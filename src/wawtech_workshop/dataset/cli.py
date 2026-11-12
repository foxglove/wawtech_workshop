"""Command-line entry point for ``build-dataset``."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from wawtech_workshop.dataset.convert import EpisodeInput, convert_episodes
from wawtech_workshop.dataset.foxglove_api import (
    ApiEpisode,
    SelectionReport,
    download_episode,
    list_version_episodes,
    select_episodes,
)


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
        help="Outcome property values to keep (default: success).",
    )
    parser.add_argument(
        "--outcome-key", default="outcome", help="Episode property used as the outcome."
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
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Convert episodes even when the API reports missing recordings.",
    )
    parser.add_argument("--output", type=Path, help="LeRobot dataset directory to create.")
    parser.add_argument("--repo-id", help="LeRobot repo id stored in the dataset.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Topics to download and convert. Default: detect state, action, and image topics.",
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
        print(f"  {episode.id} device={episode.device_id} start={episode.start} end={episode.end}")
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
        episodes = list_version_episodes(args.dataset_id, args.version)
        report = select_episodes(
            episodes,
            outcome_key=args.outcome_key,
            keep_values=tuple(args.keep_outcomes),
            allow_missing=args.allow_missing,
        )
        if args.dry_run:
            _print_report(report)
            return
        if args.output is None or args.repo_id is None:
            raise ValueError("--output and --repo-id are required unless --dry-run is set.")
        if args.project_dir is not None:
            args.project_dir.mkdir(parents=True, exist_ok=True)
            _download_and_convert(args, report.kept, args.project_dir)
            return
        with tempfile.TemporaryDirectory(prefix="wawtech-episodes-") as temporary:
            _download_and_convert(args, report.kept, Path(temporary))
        return
    if args.dry_run:
        raise ValueError("--dry-run applies to --dataset-id, not --mcap-dir.")
    if args.output is None or args.repo_id is None:
        raise ValueError("--output and --repo-id are required.")
    paths = sorted(path for path in args.mcap_dir.glob("*.mcap") if path.is_file())
    if not paths:
        raise ValueError(f"No *.mcap files in {args.mcap_dir}.")
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


def _download_and_convert(
    args: argparse.Namespace, episodes: list[ApiEpisode], directory: Path
) -> None:
    sources: list[EpisodeInput] = []
    for index, episode in enumerate(episodes):
        dest = directory / f"{episode.id}.mcap"
        download_episode(episode, dest, topics=args.topics)
        sources.append(EpisodeInput(path=dest, task=episode.task, episode_index=index))
    if not sources:
        raise ValueError("No episodes left to download after outcome filtering.")
    convert_episodes(
        sources,
        args.output,
        args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        max_gap_s=args.max_gap_s,
        topics=args.topics,
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
