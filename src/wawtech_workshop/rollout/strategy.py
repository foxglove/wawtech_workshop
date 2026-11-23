"""``--strategy.type=mcap``: a ``lerobot-rollout`` strategy that records MCAP.

``lerobot-rollout`` has no MCAP output, so this registers a strategy next to the
built-ins. It runs the same autonomous loop as ``base`` (policy, interpolation,
safety holds, interactive mode) and logs every control tick into a
:class:`~wawtech_workshop.recorder.session.RecordingSession`, so a rollout file
has the topics, H.264 video, metadata and 3D-arm topics of a teleop episode and
streams live to Foxglove the same way.

Rollout files are named ``rollout_NNNN_<session>.mcap`` and carry no ``outcome``:
they are labelled in Foxglove afterwards. They must stay out of ``data/``
(Foxlet uploads everything under ``data/episodes``) and out of any folder
``build-dataset --mcap-dir`` reads, so the output folder is checked at parse time.

LeRobot finds the strategy class by stripping ``Config`` from the config class
name (``McapStrategyConfig`` -> ``McapStrategy``) in this same module.
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import lerobot
from lerobot.processor import TransitionKey, transition_to_robot_action
from lerobot.rollout.configs import RolloutStrategyConfig
from lerobot.rollout.strategies.base import BaseStrategy

from wawtech_workshop.recorder.loop import as_rgb, camera_names_for
from wawtech_workshop.recorder.session import (
    EPISODE_GLOB,
    ROLLOUT_GLOB,
    ROLLOUT_PREFIX,
    RecordingSession,
    next_episode_index,
)
from wawtech_workshop.recorder.stream import start_stream_server

if TYPE_CHECKING:
    from lerobot.lerobot_types import EnvTransition
    from lerobot.rollout.context import RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/rollouts")
# Folders rollouts must never land in, relative to the working directory. ``data`` covers
# ``data/episodes``, the folder Foxlet watches and the one ``build-dataset`` reads.
PROTECTED_DIRS = (Path("data/episodes"), Path("data"))
STREAM_HINT = (
    "Is another recorder, rollout-episode or lerobot-rollout using it? Pass "
    "--strategy.stream_port=<port>, or --strategy.stream=false to record without streaming."
)


def _is_inside(path: Path, root: Path) -> bool:
    """Whether ``path`` is ``root`` or below it, following symlinks and letter case."""
    root = root.resolve()
    for candidate in (path, *path.parents):
        if candidate == root:
            return True
        try:
            if candidate.exists() and root.exists() and os.path.samefile(candidate, root):
                return True
        except OSError:
            continue
    return False


def check_output_dir(output_dir: Path, *, cwd: Path | None = None) -> None:
    """Refuse an output folder that would let a rollout reach training data or Foxlet.

    Raises ``ValueError`` when ``output_dir`` is, or is inside, ``data`` (and so
    ``data/episodes``) under ``cwd``, or when it already holds ``episode_*.mcap`` files.
    """
    base = cwd or Path.cwd()
    resolved = (base / output_dir).resolve()
    for protected in PROTECTED_DIRS:
        if _is_inside(resolved, base / protected):
            raise ValueError(
                f"--strategy.output_dir={str(output_dir)!r} is inside {str(protected)!r}. Rollouts "
                "must stay out of the training episodes and out of Foxlet's watched folder "
                f"(data/episodes); use a folder like {str(DEFAULT_OUTPUT_DIR)!r}."
            )
    if resolved.is_dir():
        episodes = sorted(path.name for path in resolved.glob(EPISODE_GLOB))
        if episodes:
            raise ValueError(
                f"--strategy.output_dir={str(output_dir)!r} already holds teleop episodes "
                f"({episodes[0]}{' ...' if len(episodes) > 1 else ''}). Rollouts must stay out of "
                "the training episodes and out of Foxlet's watched folder; use a separate folder "
                f"like {str(DEFAULT_OUTPUT_DIR)!r}."
            )


def check_port_free(host: str, port: int) -> None:
    """Raise ``RuntimeError`` when ``host:port`` cannot be bound. Port 0 always passes.

    The arm connects while the rollout context is built, before the strategy starts
    its server, so a taken port is caught here, at parse time, before any hardware.
    """
    if port == 0:
        return
    try:
        family, kind, proto, _name, address = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )[0]
        with socket.socket(family, kind, proto) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(address)
    except (OSError, IndexError) as exc:
        reason = "already in use" if getattr(exc, "errno", None) == errno.EADDRINUSE else str(exc)
        raise RuntimeError(f"Cannot stream on {host}:{port} ({reason}). {STREAM_HINT}") from exc


@RolloutStrategyConfig.register_subclass("mcap")
@dataclass
class McapStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout that records each run to an MCAP file and streams it live."""

    supports_interactive: ClassVar[bool] = True
    dataset_mode: ClassVar[str] = "none"

    # Rollout files land here as rollout_NNNN_<session>.mcap. Never under data/.
    output_dir: Path = DEFAULT_OUTPUT_DIR
    # Stream the same data live to the Foxglove app (replaces --display_data).
    stream: bool = True
    stream_host: str = "127.0.0.1"
    stream_port: int = 8765
    # Suffix on file names. Default: the start time.
    session_id: str | None = None
    # Robot type written to the file. Default: the robot's name.
    robot_type: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        check_output_dir(self.output_dir)
        if self.stream:
            check_port_free(self.stream_host, self.stream_port)


class McapStrategy(BaseStrategy):
    """``base`` plus an MCAP recording per ``run()`` and a live Foxglove stream."""

    config: McapStrategyConfig

    def __init__(self, config: McapStrategyConfig) -> None:
        super().__init__(config)
        self._session: RecordingSession | None = None
        self._server: Any = None

    def setup(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        if cfg.display_data:
            raise ValueError(
                "--display_data cannot be combined with --strategy.type=mcap: the strategy "
                "streams to Foxglove itself (--strategy.stream, on by default). Drop "
                "--display_data and --display_mode."
            )
        if not cfg.task.strip():
            raise ValueError("--task is required with --strategy.type=mcap.")
        if float(cfg.fps) != int(cfg.fps) or cfg.fps <= 0:
            raise ValueError(f"--fps must be a positive whole number to record, got {cfg.fps}.")
        robot = ctx.hardware.robot_wrapper.inner
        state_names = tuple(ctx.data.ordered_action_keys)
        observed = set(robot.observation_features)
        missing = [name for name in state_names if name not in observed]
        if missing:
            raise ValueError(f"The robot observation has no values for {missing}.")
        camera_names = camera_names_for(robot)

        self._session = RecordingSession(
            output_dir=self.config.output_dir,
            fps=int(cfg.fps),
            robot_type=self.config.robot_type or robot.name,
            state_names=state_names,
            camera_names=camera_names,
            lerobot_version=lerobot.__version__,
            session_id=self.config.session_id,
            file_prefix=ROLLOUT_PREFIX,
        )
        try:
            if self.config.stream:
                self._server = start_stream_server(
                    self._session.context,
                    self.config.stream_host,
                    self.config.stream_port,
                    name="wawtech rollout",
                    hint=STREAM_HINT,
                )
            super().setup(ctx)
        except BaseException:
            self._close_recording()
            raise
        print(f"Recording rollouts to {self.config.output_dir} as {ROLLOUT_PREFIX}_NNNN_*.mcap.")
        print(f"Joints: {', '.join(state_names)}. Cameras: {', '.join(camera_names)}.")

    def run(self, ctx: RolloutContext) -> None:
        """One file per call: interactive mode calls this once per /start segment."""
        session = self._require_session()
        session.start_episode(
            task=ctx.runtime.cfg.task,
            episode_index=next_episode_index(session.output_dir, ROLLOUT_GLOB),
        )
        end_reason = "error"
        failing = True
        try:
            super().run(ctx)
            # LeRobot's signal handler only sets shutdown_event, so a Ctrl-C or SIGTERM
            # makes run() return normally; without the event it ran out of --duration.
            end_reason = "stopped" if ctx.runtime.shutdown_event.is_set() else "completed"
            failing = False
        except KeyboardInterrupt:
            end_reason = "interrupted"
            raise
        finally:
            self._close_episode(ctx, end_reason, failing=failing)

    def teardown(self, ctx: RolloutContext) -> None:
        try:
            self._close_recording()
        finally:
            super().teardown(ctx)

    def _log_telemetry(
        self,
        obs_processed: dict | None,
        step: EnvTransition | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log one tick: the processed observation and the action that was sent.

        ``step`` is None while warming up or waiting, and has no action on a held tick.
        With ``--interpolation_multiplier`` above 1 the loop sends several commands per
        policy action; only the tick that emits the policy's own action is logged, as
        the dataset strategies do, so the file's rate stays ``--fps``.
        """
        session = self._session
        if session is None or obs_processed is None or step is None:
            return
        if step.get(TransitionKey.ACTION) is None:
            return
        if self._interpolator is not None and not self._interpolator.emitted_policy_action:
            return
        action = transition_to_robot_action(step)
        session.log_step(
            time.time_ns(),
            state=obs_processed,
            action=action,
            images={name: as_rgb(obs_processed[name]) for name in session.camera_names},
        )

    def _require_session(self) -> RecordingSession:
        if self._session is None:
            raise RuntimeError("McapStrategy: setup() has not run, or the strategy was torn down.")
        return self._session

    def _close_episode(self, ctx: RolloutContext, end_reason: str, *, failing: bool) -> None:
        """Save the open episode, or drop it when no step was logged.

        Runs from ``finally`` in :meth:`run`: while an exception is propagating, a
        failure here is logged instead of replacing it.
        """
        session = self._session
        if session is None or not session.recording:
            return
        try:
            if session.episode_steps == 0:
                path = session.discard_episode()
                print(f"No steps were logged; discarded {path.name}.")
                return
            result = session.finish_episode(None, self._extra_metadata(ctx, end_reason))
            print(
                f"Saved rollout {result.path} ({end_reason}): {result.steps} steps, "
                f"{result.duration_s:.1f} s."
            )
        except Exception:
            if not failing:
                raise
            logger.exception("Could not save the rollout file after the rollout failed.")
        finally:
            if session.recording:
                session.discard_episode()

    @staticmethod
    def _extra_metadata(ctx: RolloutContext, end_reason: str) -> dict[str, str]:
        cfg = ctx.runtime.cfg
        extra = {"source": "policy", "end_reason": end_reason}
        if cfg.policy is not None:
            if cfg.policy.pretrained_path is not None:
                extra["policy_path"] = str(cfg.policy.pretrained_path)
            extra["policy_type"] = str(cfg.policy.type)
        if cfg.device is not None:
            extra["device"] = str(cfg.device)
        return extra

    def _close_recording(self) -> None:
        """Stop the server and close the session. Safe to call twice."""
        server, self._server = self._server, None
        session, self._session = self._session, None
        try:
            if server is not None:
                server.stop()
        finally:
            if session is not None:
                session.close()
