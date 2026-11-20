"""Command-line entry point for ``build-dataset``."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from mcap.reader import make_reader

from wawtech_workshop.dataset.convert import EpisodeInput, convert_episodes
from wawtech_workshop.dataset.foxglove_source import (
    SelectionReport,
    default_topics,
    fetch_episodes,
)
from wawtech_workshop.mcap_schema import RECORDING_METADATA_NAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-dataset",
        description=(
            "Convert Foxglove episode MCAPs, or a dataset version, into a LeRobot v3 dataset."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-id", help="Foxglove dataset id. Requires FOXGLOVE_API_TOKEN.")
    source.add_argument(
        "--mcap-dir", type=Path, help="Directory of episode *.mcap files. Skips the API."
    )
    parser.add_argument(
        "--version", type=int, help="Dataset version number. Required with --dataset-id."
    )
    parser.add_argument(
        "--keep-outcomes",
        nargs="+",
        default=["success"],
        help=(
            "Outcomes to keep (default: success). With --dataset-id, the episode property; "
            "with --mcap-dir, the outcome record-episode wrote."
        ),
    )
    parser.add_argument(
        "--outcome-key", default="outcome", help="Episode property used as the outcome."
    )
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help=(
            "With --dataset-id, also keep episodes that have no outcome property, for an "
            "org without one. Curate the dataset version by hand first."
        ),
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Directory for MCAPs downloaded from Foxglove. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selection report and the episodes that would be downloaded.",
    )
    parser.add_argument("--output", type=Path, help="LeRobot dataset directory to create.")
    parser.add_argument("--repo-id", help="LeRobot repo id stored in the dataset.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help=(
            "Topics to download and convert. Default: with --mcap-dir, detect state, action, "
            "and image topics; with --dataset-id, state, action, and one image topic per "
            "--cameras name."
        ),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist"],
        help="Cameras to download with --dataset-id when --topics is unset (default: wrist).",
    )
    parser.add_argument("--robot-type", default="rebot_b601_follower")
    parser.add_argument("--max-gap-s", type=float, default=0.25)
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Task used when an MCAP has no episode metadata task and, in API mode, "
            "the episode has no task property."
        ),
    )
    return parser


def _print_report(report: SelectionReport) -> None:
    print(f"kept: {len(report.kept)}")
    for episode in report.kept:
        start = episode.start.isoformat()
        end = episode.end.isoformat()
        print(f"  {episode.id} start={start} end={end}")
    print(f"rejected: {len(report.rejected_ids)} {', '.join(report.rejected_ids)}".rstrip())
    print(
        f"unlabeled: {report.unlabeled_count} (excluded, missing outcome) "
        f"{', '.join(report.unlabeled_ids)}".rstrip()
    )
    print("dry-run: not downloading")


def _run(argv: list[str] | None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dataset_id is not None:
        if args.version is None:
            raise ValueError("--version is required with --dataset-id.")
        if not os.environ.get("FOXGLOVE_API_TOKEN"):
            raise RuntimeError("FOXGLOVE_API_TOKEN is not set.")
        topics = args.topics or default_topics(args.cameras)
        if args.dry_run:
            _print_report(_fetch(args, topics, None))
            return
        if args.output is None or args.repo_id is None:
            raise ValueError("--output and --repo-id are required unless --dry-run is set.")
        if args.project_dir is not None:
            _download_and_convert(args, topics, args.project_dir)
            return
        with tempfile.TemporaryDirectory(prefix="wawtech-episodes-") as temporary:
            _download_and_convert(args, topics, Path(temporary))
        return
    if args.dry_run:
        raise ValueError("--dry-run applies to --dataset-id, not --mcap-dir.")
    if args.output is None or args.repo_id is None:
        raise ValueError("--output and --repo-id are required.")
    paths = sorted(path for path in args.mcap_dir.glob("*.mcap") if path.is_file())
    if not paths:
        raise ValueError(f"No *.mcap files in {args.mcap_dir}.")
    paths = _keep_recorded_outcomes(paths, set(args.keep_outcomes))
    convert_episodes(
        paths,
        args.output,
        args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        max_gap_s=args.max_gap_s,
        topics=args.topics,
        default_task=args.task,
    )


def _keep_recorded_outcomes(paths: list[Path], keep: set[str]) -> list[Path]:
    """Drop files whose ``recording`` record has an outcome outside ``keep``.

    Files without that record (not written by ``record-episode``) are kept.
    """
    kept: list[Path] = []
    for path in paths:
        outcome = _recorded_outcome(path)
        if outcome is not None and outcome not in keep:
            print(f"skipping {path.name}: outcome {outcome!r}")
            continue
        kept.append(path)
    if not kept:
        raise ValueError(f"No episodes left after keeping outcomes {sorted(keep)}.")
    return kept


def _recorded_outcome(path: Path) -> str | None:
    with path.open("rb") as handle:
        for record in make_reader(handle).iter_metadata():
            if record.name == RECORDING_METADATA_NAME:
                return record.metadata.get("outcome")
    return None


def _fetch(args: argparse.Namespace, topics: list[str], directory: Path | None) -> SelectionReport:
    return fetch_episodes(
        args.dataset_id,
        args.version,
        topics,
        download_dir=directory,
        outcome_key=args.outcome_key,
        keep_values=tuple(args.keep_outcomes),
        keep_unlabeled=args.keep_unlabeled,
    )


def _download_and_convert(args: argparse.Namespace, topics: list[str], directory: Path) -> None:
    report = _fetch(args, topics, directory)
    sources = [
        EpisodeInput(path=episode.path, task=episode.task, episode_index=index)
        for index, episode in enumerate(report.kept)
        if episode.path is not None
    ]
    if not sources:
        raise ValueError("No episodes left to download after outcome filtering.")
    convert_episodes(
        sources,
        args.output,
        args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        max_gap_s=args.max_gap_s,
        topics=topics,
        default_task=args.task,
    )


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0]


def main(argv: list[str] | None = None) -> None:
    """Run the converter. Exits non-zero and prints one error line on failure."""
    try:
        _run(argv)
    except Exception as exc:
        print(f"error: {_one_line(exc)}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
