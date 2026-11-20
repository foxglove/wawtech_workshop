"""``upload-episodes``: put recorded episode MCAPs into Foxglove, optionally into a dataset.

Steps, each one optional after the first:

1. Upload every ``episode_*.mcap`` with ``Client.upload_data``. The recording
   key is the file stem, so uploading the same file again is de-duplicated.
2. With ``--dataset-id``: wait for the imports to finish, create one Foxglove
   episode per recording with the ``task`` and ``outcome`` episode properties
   (read from the file's ``episode`` and ``recording`` metadata), and stage
   those episodes in the dataset's editable version.
3. With ``--commit``: commit the dataset version so ``build-dataset
   --dataset-id ... --version N`` can read it.

``task`` and ``outcome`` are episode custom properties, and only properties the
org defines can be set. Before staging, the org's episode properties are listed
and any that are missing are left off with a warning, so an org that is not set
up yet still works:

* without ``outcome``, only ``success`` episodes are uploaded (unless
  ``--outcomes`` says otherwise), since nothing would mark the failures later;
  ``build-dataset --dataset-id`` then needs ``--keep-unlabeled``;
* without ``task``, ``build-dataset`` needs ``--task``.

``--create-properties`` creates both in the org (``task`` a string, ``outcome``
an enum of success and failure). Pass ``--no-properties`` to skip properties.

Needs ``FOXGLOVE_API_TOKEN`` (``FOXGLOVE_API_URL`` optional). The token is never
printed. ``--dry-run`` reads the files and prints the plan without any API call.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcap.reader import make_reader

from wawtech_workshop.dataset.foxglove_source import make_client
from wawtech_workshop.mcap_schema import (
    EPISODE_METADATA_NAME,
    RECORDING_METADATA_NAME,
    EpisodeMetadata,
)
from wawtech_workshop.recorder.session import EPISODE_GLOB

DEFAULT_DEVICE_NAME = "rebot-b601"
_FAILED_STATUSES = {"failed", "error"}
OUTCOME_VALUES = ["success", "failure"]


@dataclass(frozen=True)
class LocalEpisode:
    path: Path
    key: str
    task: str
    outcome: str | None
    episode_index: int


def read_local_episode(path: Path) -> LocalEpisode:
    episode: EpisodeMetadata | None = None
    recording: dict[str, str] = {}
    with path.open("rb") as handle:
        for record in make_reader(handle).iter_metadata():
            if record.name == EPISODE_METADATA_NAME:
                episode = EpisodeMetadata.from_mcap_metadata(record.metadata)
            elif record.name == RECORDING_METADATA_NAME:
                recording = dict(record.metadata)
    if episode is None:
        raise ValueError(f"{path} has no {EPISODE_METADATA_NAME!r} metadata; was it recorded here?")
    return LocalEpisode(
        path=path,
        key=path.stem,
        task=episode.task,
        outcome=recording.get("outcome"),
        episode_index=episode.episode_index,
    )


def find_episodes(paths: Sequence[Path]) -> list[LocalEpisode]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob(EPISODE_GLOB)) if path.is_dir() else [path])
    if not files:
        raise ValueError(f"No {EPISODE_GLOB} files in {[str(path) for path in paths]}.")
    episodes = [read_local_episode(path) for path in files]
    return sorted(episodes, key=lambda episode: (episode.episode_index, episode.key))


def upload(
    client: Any,
    episodes: Sequence[LocalEpisode],
    *,
    device_name: str,
    project_id: str | None,
    say: Callable[[str], None] = print,
) -> None:
    for episode in episodes:
        with episode.path.open("rb") as handle:
            result = client.upload_data(
                device_name=device_name,
                key=episode.key,
                filename=episode.path.name,
                data=handle,
                project_id=project_id,
            )
        code = result.get("code")
        if code is not None and not 200 <= int(code) < 300:
            raise RuntimeError(f"Upload of {episode.path.name} failed with HTTP {code}.")
        say(f"Uploaded {episode.path.name} (key {episode.key}).")


def wait_for_recordings(
    client: Any,
    keys: Sequence[str],
    *,
    device_name: str,
    project_id: str | None,
    timeout_s: float,
    poll_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
) -> dict[str, str]:
    """Return ``{key: recording_id}`` once every key has a completed import."""
    wanted = set(keys)
    found: dict[str, str] = {}
    deadline = monotonic() + timeout_s
    while True:
        recordings = client.get_recordings(
            device_name=device_name,
            project_id=project_id,
            sort_by="created_at",
            sort_order="desc",
            limit=max(100, 2 * len(wanted)),
        )
        for recording in recordings:
            key = recording.get("key")
            if key not in wanted or key in found:
                continue
            status = str(recording.get("import_status") or "").lower()
            if status in _FAILED_STATUSES:
                raise RuntimeError(f"Foxglove failed to import recording {key}.")
            if status == "complete":
                found[key] = recording["id"]
        missing = sorted(wanted - set(found))
        if not missing:
            return found
        if monotonic() >= deadline:
            raise TimeoutError(
                f"{len(missing)} recordings were not imported within {timeout_s:.0f} s: {missing}."
            )
        say(f"Waiting for {len(missing)} imports...")
        sleep(poll_s)


def stage_in_dataset(
    client: Any,
    episodes: Sequence[LocalEpisode],
    recording_ids: dict[str, str],
    *,
    dataset_id: str,
    project_id: str,
    task_key: str | None,
    outcome_key: str | None,
) -> list[str]:
    """Create one episode per recording and stage them in the dataset. Returns episode ids."""
    definitions = []
    for episode in episodes:
        properties: dict[str, str] = {}
        if task_key:
            properties[task_key] = episode.task
        if outcome_key and episode.outcome:
            properties[outcome_key] = episode.outcome
        definition: dict[str, Any] = {"recordings": [recording_ids[episode.key]]}
        if properties:
            definition["properties"] = properties
        definitions.append(definition)
    created = client.create_episodes(project_id=project_id, episodes=definitions)
    episode_ids = [row["id"] for row in created]
    if len(episode_ids) != len(episodes):
        raise RuntimeError(
            f"Foxglove returned {len(episode_ids)} episodes for {len(episodes)} recordings."
        )
    client.update_dataset_episodes(dataset_id=dataset_id, add=episode_ids)
    return episode_ids


def episode_property_keys(client: Any, say: Callable[[str], None] = print) -> set[str] | None:
    """Keys of the org's episode custom properties, or None when they cannot be listed."""
    try:
        properties = client.get_custom_properties(resource_type="episode")
    except Exception as exc:  # noqa: BLE001  (any API failure means "unknown")
        say(f"Warning: could not list episode properties ({exc}); creating episodes without any.")
        return None
    return {str(item["key"]) for item in properties}


def create_episode_properties(
    client: Any,
    *,
    task_key: str,
    outcome_key: str,
    existing: set[str],
    say: Callable[[str], None] = print,
) -> set[str]:
    """Create the task and outcome episode properties the org is missing."""
    wanted = [
        (task_key, {"label": "Task", "value_type": "string"}),
        (outcome_key, {"label": "Outcome", "value_type": "enum", "enum_values": OUTCOME_VALUES}),
    ]
    created = set(existing)
    for key, options in wanted:
        if key in created:
            continue
        client.create_custom_property(key=key, resource_type="episode", **options)
        say(f"Created episode property {key!r}.")
        created.add(key)
    return created


def resolve_property_keys(
    keys: set[str] | None,
    *,
    task_key: str,
    outcome_key: str,
    say: Callable[[str], None] = print,
) -> tuple[str | None, str | None]:
    """Drop the property keys the org does not define, with one line on what to set up."""
    available = keys or set()
    missing = [key for key in (task_key, outcome_key) if key not in available]
    if keys is not None and missing:
        say(
            f"The Foxglove org has no episode properties {missing}; episodes are created "
            "without them. Run again with --create-properties once the org is set up, or "
            "add them in the org settings."
        )
    return (
        task_key if task_key in available else None,
        outcome_key if outcome_key in available else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="upload-episodes",
        description="Upload recorded episode MCAPs to Foxglove and stage them in a dataset.",
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Episode MCAP files or directories of them."
    )
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME)
    parser.add_argument(
        "--project-id", help="Foxglove project. Defaults to the dataset's project when set."
    )
    parser.add_argument("--dataset-id", help="Stage the uploaded episodes in this dataset.")
    parser.add_argument(
        "--commit", action="store_true", help="Commit the dataset version after staging."
    )
    parser.add_argument("--commit-description", default=None)
    parser.add_argument("--task-key", default="task", help="Episode property for the task.")
    parser.add_argument(
        "--outcome-key", default="outcome", help="Episode property for the outcome."
    )
    parser.add_argument(
        "--no-properties", action="store_true", help="Create episodes without properties."
    )
    parser.add_argument(
        "--create-properties",
        action="store_true",
        help="Create the task and outcome episode properties in the org if they are missing.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=None,
        help="Only upload episodes with these outcomes (default: all).",
    )
    parser.add_argument("--import-timeout-s", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None, *, client: Any = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.commit and not args.dataset_id:
        raise SystemExit("--commit needs --dataset-id.")
    if args.create_properties and args.no_properties:
        raise SystemExit("--create-properties and --no-properties contradict each other.")
    episodes = _filter_outcomes(find_episodes(args.paths), args.outcomes)
    for episode in episodes:
        print(
            f"{episode.path.name}: episode {episode.episode_index}, "
            f"outcome {episode.outcome or '(none)'}, task {episode.task!r}"
        )
    if not episodes:
        print("Nothing to upload.")
        return 0
    if args.dry_run:
        target = f"dataset {args.dataset_id}" if args.dataset_id else "recordings only"
        print(
            f"Dry run: would upload {len(episodes)} files "
            f"as device {args.device_name!r} ({target})."
        )
        return 0
    client = client or make_client()
    project_id = args.project_id
    task_key: str | None = None
    outcome_key: str | None = None
    if args.dataset_id:
        if project_id is None:
            project_id = client.get_dataset(dataset_id=args.dataset_id)["project_id"]
        if not args.no_properties:
            keys = episode_property_keys(client)
            if args.create_properties and keys is not None:
                keys = create_episode_properties(
                    client, task_key=args.task_key, outcome_key=args.outcome_key, existing=keys
                )
            task_key, outcome_key = resolve_property_keys(
                keys, task_key=args.task_key, outcome_key=args.outcome_key
            )
            if outcome_key is None and args.outcomes is None:
                episodes = _filter_outcomes(episodes, ["success"])
                print(
                    f"Without an outcome property, uploading only the {len(episodes)} success "
                    "episodes. Build with: build-dataset --dataset-id ... --keep-unlabeled"
                )
                if not episodes:
                    print("Nothing to upload.")
                    return 0
            if task_key is None:
                print("Without a task property, pass --task to build-dataset.")
    upload(client, episodes, device_name=args.device_name, project_id=project_id)
    if not args.dataset_id:
        return 0
    recording_ids = wait_for_recordings(
        client,
        [episode.key for episode in episodes],
        device_name=args.device_name,
        project_id=project_id,
        timeout_s=args.import_timeout_s,
    )
    episode_ids = stage_in_dataset(
        client,
        episodes,
        recording_ids,
        dataset_id=args.dataset_id,
        project_id=project_id,
        task_key=task_key,
        outcome_key=outcome_key,
    )
    print(f"Staged {len(episode_ids)} episodes in dataset {args.dataset_id}.")
    if args.commit:
        kwargs = {}
        if args.commit_description is not None:
            kwargs["description"] = args.commit_description
        result = client.commit_dataset(dataset_id=args.dataset_id, **kwargs)
        version = result["committed"]["version_number"]
        print(f"Committed dataset version {version}.")
    else:
        print("Review the staged episodes in Foxglove, then commit the version there.")
    return 0


def _filter_outcomes(
    episodes: list[LocalEpisode], outcomes: Sequence[str] | None
) -> list[LocalEpisode]:
    if outcomes is None:
        return episodes
    wanted = set(outcomes)
    return [episode for episode in episodes if episode.outcome in wanted]


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
