"""Rewrite JSON ``foxglove.CompressedVideo`` channels in MCAP files as protobuf.

Older recordings logged the camera through a generic channel, so every frame is a
JSON message with base64 ``data``. Foxglove semantic search only indexes
``foxglove.CompressedVideo`` when it is protobuf, so those recordings fail indexing.
This tool converts them without re-recording::

    uv run convert-video-protobuf data/episodes data/episodes_pb

Each ``<stem>.mcap`` in the input folder becomes ``<stem>_pb.mcap`` in the output
folder. The inputs are only read. Every schema, channel, message, attachment and
metadata record is copied unchanged, except the JSON ``foxglove.CompressedVideo``
channels: their messages are rebuilt as protobuf with the same timestamp,
``frame_id``, ``format`` and ``data`` bytes, on a channel with the same topic, and
keep their ``log_time``, ``publish_time`` and ``sequence``.

Each output is verified before it is moved into place (see :func:`verify`). Nothing
is uploaded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from foxglove.messages import CompressedVideo, Timestamp
from mcap.reader import make_reader
from mcap.writer import Writer

from wawtech_workshop.compressed_video import load_compressed_video, parse_compressed_video
from wawtech_workshop.dataset.video import decode_h264_messages
from wawtech_workshop.mcap_schema import COMPRESSED_VIDEO_SCHEMA_NAME

OUTPUT_SUFFIX = "_pb"
DECODE_CHECK_FRAMES = 3


@dataclass
class FileReport:
    source: Path
    output: Path | None
    status: str  # "ok", "skipped" or "FAILED"
    detail: str
    converted: dict[str, int]  # topic -> messages rewritten
    total_messages: int = 0


def output_path(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{source.stem}{OUTPUT_SUFFIX}.mcap"


def _is_json_video(schema, channel) -> bool:
    return (
        schema is not None
        and schema.name == COMPRESSED_VIDEO_SCHEMA_NAME
        and channel.message_encoding == "json"
    )


def convert_file(source: Path, dest: Path) -> dict[str, int]:
    """Write ``dest`` from ``source``. Returns messages rewritten per topic."""
    pb_schema = CompressedVideo.get_schema()
    converted: Counter[str] = Counter()
    with source.open("rb") as src, dest.open("wb") as out:
        reader = make_reader(src)
        header = reader.get_header()
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("file has no summary section, so channels cannot be listed")
        schemas = summary.schemas
        channels = summary.channels
        json_video = {
            cid
            for cid, channel in channels.items()
            if _is_json_video(schemas.get(channel.schema_id), channel)
        }
        # A schema is dropped only if every channel using it is being converted.
        kept_schema_ids = {
            channel.schema_id
            for cid, channel in channels.items()
            if cid not in json_video and channel.schema_id != 0
        }

        writer = Writer(out)
        writer.start(
            profile=header.profile,
            library=f"{header.library}; video json->protobuf (wawtech_workshop)",
        )
        schema_ids = {
            sid: writer.register_schema(name=s.name, encoding=s.encoding, data=s.data)
            for sid, s in sorted(schemas.items())
            if sid in kept_schema_ids
        }
        pb_schema_id = (
            writer.register_schema(
                name=pb_schema.name, encoding=pb_schema.encoding, data=pb_schema.data
            )
            if json_video
            else 0
        )
        channel_ids: dict[int, int] = {}
        for cid, channel in sorted(channels.items()):
            if cid in json_video:
                channel_ids[cid] = writer.register_channel(
                    topic=channel.topic,
                    message_encoding="protobuf",
                    schema_id=pb_schema_id,
                    metadata=dict(channel.metadata),
                )
            else:
                channel_ids[cid] = writer.register_channel(
                    topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    schema_id=schema_ids.get(channel.schema_id, 0),
                    metadata=dict(channel.metadata),
                )

        for _schema, channel, message in reader.iter_messages(log_time_order=True):
            data = message.data
            if channel.id in json_video:
                video = load_compressed_video("json", data)
                stamp = video["timestamp"]
                data = CompressedVideo(
                    timestamp=Timestamp(sec=int(stamp["sec"]), nsec=int(stamp["nsec"])),
                    frame_id=str(video["frame_id"]),
                    data=video["data"],
                    format=str(video["format"]),
                ).encode()
                converted[channel.topic] += 1
            writer.add_message(
                channel_id=channel_ids[channel.id],
                log_time=message.log_time,
                data=data,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )
        for attachment in reader.iter_attachments():
            writer.add_attachment(
                create_time=attachment.create_time,
                log_time=attachment.log_time,
                name=attachment.name,
                media_type=attachment.media_type,
                data=attachment.data,
            )
        for record in reader.iter_metadata():
            writer.add_metadata(name=record.name, data=dict(record.metadata))
        writer.finish()
    return dict(converted)


def verify(source: Path, dest: Path) -> tuple[int, int]:
    """Check ``dest`` against ``source``. Returns ``(total_messages, frames_decoded)``.

    Raises ``ValueError`` on the first problem:

    * every video channel in ``dest`` has ``message_encoding`` ``protobuf`` and the
      ``foxglove.CompressedVideo`` protobuf schema, and none is left as JSON;
    * every topic has the same message count as in ``source``;
    * the video payload of every message is identical to ``source`` (timestamp,
      ``frame_id``, ``format``, ``data`` bytes) and so are ``log_time`` and
      ``publish_time``;
    * attachments and metadata records are equal;
    * the first few frames of each video topic decode to images.
    """
    with source.open("rb") as src_handle, dest.open("rb") as dst_handle:
        src, dst = make_reader(src_handle), make_reader(dst_handle)
        src_summary, dst_summary = src.get_summary(), dst.get_summary()
        if src_summary is None or dst_summary is None:
            raise ValueError("missing summary section")

        def by_topic(summary):
            return {
                channel.topic: (summary.schemas.get(channel.schema_id), channel)
                for channel in summary.channels.values()
            }

        src_topics, dst_topics = by_topic(src_summary), by_topic(dst_summary)
        if set(src_topics) != set(dst_topics):
            raise ValueError(f"topics differ: {sorted(set(src_topics) ^ set(dst_topics))}")
        video_topics = []
        for topic, (schema, channel) in src_topics.items():
            new_schema, new_channel = dst_topics[topic]
            if _is_json_video(schema, channel):
                video_topics.append(topic)
                if new_channel.message_encoding != "protobuf":
                    raise ValueError(
                        f"{topic} message_encoding is {new_channel.message_encoding!r}, "
                        "expected 'protobuf'"
                    )
                if (
                    new_schema is None
                    or new_schema.name != COMPRESSED_VIDEO_SCHEMA_NAME
                    or new_schema.encoding != "protobuf"
                ):
                    raise ValueError(f"{topic} does not use the protobuf CompressedVideo schema")
            elif (new_channel.message_encoding, getattr(new_schema, "name", None)) != (
                channel.message_encoding,
                getattr(schema, "name", None),
            ):
                raise ValueError(f"{topic} encoding or schema changed")
        if not video_topics:
            raise ValueError("no JSON foxglove.CompressedVideo channel found to convert")

        total = 0
        counts: Counter[str] = Counter()
        previews: dict[str, list[tuple[int, dict]]] = {topic: [] for topic in video_topics}
        src_messages = src.iter_messages(log_time_order=True)
        dst_messages = dst.iter_messages(log_time_order=True)
        for (_s, s_channel, s_message), (_d, d_channel, d_message) in zip(
            src_messages, dst_messages, strict=False
        ):
            total += 1
            counts[s_channel.topic] += 1
            if s_channel.topic != d_channel.topic:
                raise ValueError(f"message {total}: topic order differs")
            if (s_message.log_time, s_message.publish_time) != (
                d_message.log_time,
                d_message.publish_time,
            ):
                raise ValueError(f"message {total} on {s_channel.topic}: times differ")
            if s_channel.topic in video_topics:
                old = load_compressed_video("json", s_message.data)
                new = parse_compressed_video(d_message.data)
                if old != new:
                    raise ValueError(f"message {total} on {s_channel.topic}: video payload differs")
                if len(previews[s_channel.topic]) < DECODE_CHECK_FRAMES:
                    previews[s_channel.topic].append((d_message.log_time, new))
            elif s_message.data != d_message.data:
                raise ValueError(f"message {total} on {s_channel.topic}: data differs")
        # zip stops at the shorter stream, so compare counts from the summaries too.
        for topic, (_, channel) in src_topics.items():
            expected = src_summary.statistics.channel_message_counts.get(channel.id, 0)
            actual = dst_summary.statistics.channel_message_counts.get(dst_topics[topic][1].id, 0)
            if not (expected == actual == counts[topic]):
                raise ValueError(
                    f"{topic} message count: input {expected}, output {actual}, "
                    f"compared {counts[topic]}"
                )

        src_attachments = [
            (a.name, a.media_type, a.create_time, a.log_time, a.data)
            for a in src.iter_attachments()
        ]
        dst_attachments = [
            (a.name, a.media_type, a.create_time, a.log_time, a.data)
            for a in dst.iter_attachments()
        ]
        if src_attachments != dst_attachments:
            raise ValueError("attachments differ")
        src_meta = [(m.name, dict(m.metadata)) for m in src.iter_metadata()]
        dst_meta = [(m.name, dict(m.metadata)) for m in dst.iter_metadata()]
        if sorted(src_meta, key=lambda m: m[0]) != sorted(dst_meta, key=lambda m: m[0]):
            raise ValueError("metadata records differ")

    decoded = 0
    for topic, frames in previews.items():
        images = decode_h264_messages(frames)
        if len(images) != len(frames):
            raise ValueError(f"{topic}: decoded {len(images)} of {len(frames)} first frames")
        decoded += len(images)
    return total, decoded


def convert_one(source: Path, output_dir: Path) -> FileReport:
    dest = output_path(source, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return FileReport(source, dest, "skipped", "output already exists", {})
    partial = dest.with_name(dest.name + ".partial")
    try:
        converted = convert_file(source, partial)
        total, decoded = verify(source, partial)
        partial.replace(dest)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return FileReport(source, None, "FAILED", f"{type(exc).__name__}: {exc}", {})
    topics = ", ".join(f"{topic} {count}" for topic, count in sorted(converted.items()))
    detail = f"{topics} msgs json->protobuf, {total} total, {decoded} frames decoded"
    return FileReport(source, dest, "ok", detail, converted, total)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert-video-protobuf",
        description=(
            "Rewrite JSON foxglove.CompressedVideo channels as protobuf so Foxglove "
            "semantic search can index them. Writes <stem>_pb.mcap files to a new "
            "folder; the originals are never modified."
        ),
    )
    parser.add_argument("input_dir", type=Path, help="Folder of .mcap files.")
    parser.add_argument("output_dir", type=Path, help="Folder for the <stem>_pb.mcap outputs.")
    args = parser.parse_args(argv)

    input_dir, output_dir = args.input_dir, args.output_dir
    if not input_dir.is_dir():
        parser.error(f"{input_dir} is not a folder.")
    if input_dir.resolve() == output_dir.resolve():
        parser.error("output_dir must differ from input_dir so the originals are not touched.")
    sources = sorted(input_dir.glob("*.mcap"))
    if not sources:
        parser.error(f"no .mcap files in {input_dir}.")

    reports = []
    for source in sources:
        report = convert_one(source, output_dir)
        reports.append(report)
        arrow = f" -> {report.output.name}" if report.output else ""
        print(f"{report.status:7} {source.name}{arrow}: {report.detail}", flush=True)

    ok = [r for r in reports if r.status == "ok"]
    failed = [r for r in reports if r.status == "FAILED"]
    skipped = [r for r in reports if r.status == "skipped"]
    video_messages = sum(sum(r.converted.values()) for r in ok)
    print(
        f"Total: {len(ok)} converted and verified, {len(failed)} failed, {len(skipped)} skipped; "
        f"{video_messages} video messages rewritten as protobuf, "
        f"{sum(r.total_messages for r in ok)} messages checked. Nothing was uploaded."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
