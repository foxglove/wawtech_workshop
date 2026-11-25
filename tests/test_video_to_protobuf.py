"""The protobuf video reader and the JSON -> protobuf recording converter."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from foxglove.messages import CompressedVideo, Timestamp
from mcap.reader import make_reader
from mcap.writer import Writer

from tests.synthetic import make_frame, write_synthetic_episode
from wawtech_workshop.compressed_video import load_compressed_video, parse_compressed_video
from wawtech_workshop.dataset.convert import EpisodeInput, convert_episodes
from wawtech_workshop.h264 import encode_h264_frames
from wawtech_workshop.video_to_protobuf import convert_one, main, output_path

T0 = 1_700_000_000_000_000_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("sec, nsec", [(1_700_000_000, 123_456_789), (0, 0), (5, 0)])
def test_parse_matches_sdk_encoding(sec: int, nsec: int) -> None:
    raw = CompressedVideo(
        timestamp=Timestamp(sec=sec, nsec=nsec),
        frame_id="wrist",
        data=b"\x00\x00\x01ab",
        format="h264",
    ).encode()
    assert parse_compressed_video(raw) == {
        "timestamp": {"sec": sec, "nsec": nsec},
        "frame_id": "wrist",
        "data": b"\x00\x00\x01ab",
        "format": "h264",
    }
    assert load_compressed_video("protobuf", raw)["data"] == b"\x00\x00\x01ab"


def test_load_json_decodes_base64_and_rejects_other_encodings() -> None:
    raw = json.dumps(
        {"timestamp": {"sec": 1, "nsec": 2}, "frame_id": "w", "data": "AAEC", "format": "h264"}
    ).encode()
    assert load_compressed_video("json", raw)["data"] == b"\x00\x01\x02"
    with pytest.raises(ValueError, match="encoding"):
        load_compressed_video("cdr", raw)


def _summary(path: Path):
    with path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    assert summary is not None
    return summary


def test_convert_keeps_everything_but_the_video_encoding(tmp_path: Path) -> None:
    source = tmp_path / "in" / "episode_0000_x.mcap"
    episode = write_synthetic_episode(source, n_frames=8, t0_ns=T0, camera_names=("wrist",))
    before = _sha(source)

    report = convert_one(source, tmp_path / "out")

    assert report.status == "ok", report.detail
    dest = tmp_path / "out" / "episode_0000_x_pb.mcap"
    assert report.output == dest and dest.exists()
    assert not list((tmp_path / "out").glob("*.partial"))
    assert _sha(source) == before  # the original is untouched
    assert report.converted == {"/observation/images/wrist": 8}

    old, new = _summary(source), _summary(dest)
    encodings = {c.topic: c.message_encoding for c in new.channels.values()}
    assert encodings == {
        "/observation/state": "json",
        "/action/state": "json",
        "/observation/images/wrist": "protobuf",
    }
    schemas = {c.topic: new.schemas[c.schema_id] for c in new.channels.values()}
    assert schemas["/observation/images/wrist"].name == "foxglove.CompressedVideo"
    assert schemas["/observation/images/wrist"].encoding == "protobuf"
    assert schemas["/observation/state"].encoding == "jsonschema"
    assert old.statistics.message_count == new.statistics.message_count

    with dest.open("rb") as handle:
        reader = make_reader(handle)
        metadata = {m.name: dict(m.metadata) for m in reader.iter_metadata()}
        video = [
            (m.log_time, m.publish_time, parse_compressed_video(m.data))
            for _s, _c, m in reader.iter_messages(topics=["/observation/images/wrist"])
        ]
    assert metadata["episode"]["task"] == episode.metadata.task
    assert [stamp for stamp, _, _ in video] == episode.times_ns
    assert all(stamp == published for stamp, published, _ in video)
    assert video[0][2]["frame_id"] == "wrist" and video[0][2]["format"] == "h264"
    assert video[3][2]["timestamp"] == {"sec": T0 // 10**9, "nsec": episode.times_ns[3] % 10**9}


def test_converted_file_still_builds_a_dataset(tmp_path: Path) -> None:
    source = tmp_path / "in" / "episode_0000_x.mcap"
    episode = write_synthetic_episode(source, n_frames=8, t0_ns=T0, camera_names=("wrist",))
    assert convert_one(source, tmp_path / "out").status == "ok"

    root = convert_episodes(
        [EpisodeInput(path=tmp_path / "out" / "episode_0000_x_pb.mcap")],
        tmp_path / "dataset",
        "local/pb",
        fps=episode.metadata.fps,
        robot_type=episode.metadata.robot_type,
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/pb", root=root, video_backend="pyav")
    assert dataset.meta.total_episodes == 1
    assert len(dataset) > 0


def test_copies_other_channels_attachments_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "in" / "a.mcap"
    source.parent.mkdir()
    [packet] = encode_h264_frames([make_frame(0)], 30)
    with source.open("wb") as handle:
        writer = Writer(handle)
        writer.start(profile="custom", library="test")
        video_schema = writer.register_schema(
            name="foxglove.CompressedVideo", encoding="jsonschema", data=b"{}"
        )
        raw_schema = writer.register_schema(name="other.Thing", encoding="jsonschema", data=b"{}")
        video = writer.register_channel("/cam", "json", video_schema, {"k": "v"})
        thing = writer.register_channel("/thing", "json", raw_schema)
        bare = writer.register_channel("/bare", "json", 0)
        body = json.dumps(
            {
                "timestamp": {"sec": 1, "nsec": 5},
                "frame_id": "cam",
                "data": base64.b64encode(packet).decode(),
                "format": "h264",
            }
        ).encode()
        writer.add_message(video, 10, body, 9, 4)
        writer.add_message(thing, 11, b'{"a":1}', 11, 7)
        writer.add_message(bare, 12, b"{}", 12, 8)
        writer.add_attachment(
            create_time=1, log_time=2, name="notes", media_type="text/plain", data=b"hi"
        )
        writer.add_metadata("custom", {"a": "b"})
        writer.finish()

    dest = tmp_path / "out" / "a_pb.mcap"
    dest.parent.mkdir()
    from wawtech_workshop.video_to_protobuf import convert_file, verify

    assert convert_file(source, dest) == {"/cam": 1}
    assert verify(source, dest)[0] == 3

    with dest.open("rb") as handle:
        reader = make_reader(handle)
        header = reader.get_header()
        summary = reader.get_summary()
        assert header.profile == "custom"
        by_topic = {c.topic: c for c in summary.channels.values()}
        assert dict(by_topic["/cam"].metadata) == {"k": "v"}
        assert by_topic["/bare"].schema_id == 0
        assert summary.schemas[by_topic["/thing"].schema_id].name == "other.Thing"
        messages = {
            c.topic: (m.log_time, m.publish_time, m.sequence, m.data)
            for _s, c, m in reader.iter_messages()
        }
        assert messages["/thing"] == (11, 11, 7, b'{"a":1}')
        assert messages["/cam"][:3] == (10, 9, 4)
        assert parse_compressed_video(messages["/cam"][3])["data"] == packet
        [attachment] = list(reader.iter_attachments())
        assert (attachment.name, attachment.media_type, attachment.data) == (
            "notes",
            "text/plain",
            b"hi",
        )
        assert [(m.name, dict(m.metadata)) for m in reader.iter_metadata()] == [
            ("custom", {"a": "b"})
        ]


def test_file_without_json_video_fails_and_leaves_no_output(tmp_path: Path) -> None:
    source = tmp_path / "in" / "a.mcap"
    write_synthetic_episode(source, n_frames=4, t0_ns=T0, camera_names=("wrist",))
    done = convert_one(source, tmp_path / "out")
    assert done.status == "ok"
    # Converting an already-converted file has nothing to convert.
    again = convert_one(done.output, tmp_path / "out2")
    assert again.status == "FAILED" and "no JSON" in again.detail
    assert list((tmp_path / "out2").glob("*")) == []


def test_main_skips_existing_outputs_and_refuses_the_input_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "in"
    write_synthetic_episode(src / "a.mcap", n_frames=4, t0_ns=T0, camera_names=("wrist",))
    write_synthetic_episode(src / "b.mcap", n_frames=4, t0_ns=T0, camera_names=("wrist",))

    with pytest.raises(SystemExit):
        main([str(src), str(src)])

    assert main([str(src), str(tmp_path / "out")]) == 0
    first = capsys.readouterr().out
    assert first.count("ok ") == 2 and "Total: 2 converted and verified, 0 failed" in first
    assert output_path(src / "a.mcap", tmp_path / "out").name == "a_pb.mcap"

    assert main([str(src), str(tmp_path / "out")]) == 0
    assert "2 skipped" in capsys.readouterr().out
    assert sorted(p.name for p in src.glob("*")) == ["a.mcap", "b.mcap"]
