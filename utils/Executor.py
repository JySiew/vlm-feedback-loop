from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.pipeline import PolicyProcessorPipeline

from .Skills import Skills


class ExecStatus(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class ExecResult:
    status: ExecStatus
    duration_s: float
    error: str | None = None


@dataclass(frozen=True)
class Observation:
    overhead: np.ndarray
    timestamp: float


class _PolicyEnvironmentConfig(EnvConfig):
    @property
    def gym_kwargs(self) -> dict:
        return {}


class Executor(ABC):
    def __init__(self, skills: Skills):
        if not isinstance(skills, Skills):
            raise TypeError("skills must be a Skills instance")
        self.skills = skills

    def _validate_skill(self, skill_name: str) -> None:
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name cannot be an empty string")
        if skill_name not in self.skills.get_skills():
            raise ValueError(f"Unsupported skill: {skill_name!r}")

    @abstractmethod
    def connect(self) -> None:
        """Prepare the backend for execution."""

    @abstractmethod
    def run(self, skill_name: str) -> ExecResult:
        """Execute one supported skill."""

    @abstractmethod
    def capture(self) -> Observation:
        """Return the current overhead image."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


class LocalSimulator(Executor):
    def __init__(
        self,
        before_image: str | Path | np.ndarray,
        after_images: Sequence[str | Path | np.ndarray],
        request: str,
        skills: Skills,
    ):
        super().__init__(skills)
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request cannot be an empty string")
        if isinstance(after_images, (str, Path, np.ndarray)) or not isinstance(
            after_images, Sequence
        ):
            raise TypeError("after_images must be a list or tuple of images")

        self.request = request
        self._images = [self._load_image(before_image)]
        self._images.extend(self._load_image(image) for image in after_images)
        self._index = 0
        self._closed = False
        self._batch_open = False
        self._observed_at = time.time()

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(Path(image).expanduser()) as source:
                image = np.array(source.convert("RGB"))
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Images must be file paths or uint8 RGB arrays")
        if image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
            raise ValueError("Image arrays must have shape (height, width, 3)")
        return image.copy()

    def connect(self) -> None:
        self._closed = False

    def run(self, skill_name: str) -> ExecResult:
        started = time.monotonic()
        try:
            self._validate_skill(skill_name)
            if self._closed:
                raise RuntimeError("Simulator is closed; call connect() to reopen it")
            if not self._batch_open and self._index >= len(self._images) - 1:
                raise RuntimeError("No feedback images remaining: playback is exhausted")
        except (ValueError, RuntimeError) as exc:
            return ExecResult(ExecStatus.ERROR, time.monotonic() - started, str(exc))

        if not self._batch_open:
            self._index += 1
            self._observed_at = time.time()
            self._batch_open = True
        return ExecResult(ExecStatus.COMPLETED, time.monotonic() - started)

    def capture(self) -> Observation:
        if self._closed:
            raise RuntimeError("Simulator is closed; call connect() to reopen it")
        self._batch_open = False
        return Observation(self._images[self._index].copy(), self._observed_at)

    def close(self) -> None:
        self._closed = True


class LeRobotExecutor(Executor):
    """Physical hardware executor for SO-101 follower arm and LeRobot diffusion policies."""

    CHECKPOINT_PATHS = {
        "pick_place_pencil": "/home/hoonc/outputs/diffusion/pick_place_pencil_v3/checkpoints/060000/pretrained_model_ema",
        "pick_place_red_pen": "/home/hoonc/outputs/diffusion/pick_place_red_pen_v3/checkpoints/060000/pretrained_model_ema",
        "pick_place_black_pen": "/home/hoonc/outputs/diffusion/pick_place_black_pen_v3/checkpoints/060000/pretrained_model_ema",
    }

    def __init__(
        self,
        skills: Skills,
        port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79018706-if00",
        robot_id: str = "orange_follower_arm",
        overhead_cam_idx: int = 0,
        wrist_cam_idx: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        warmup_s: int = 3,
        noise_scheduler_type: str = "DDIM",
        num_inference_steps: int = 5,
        n_action_steps: int = 16,
    ):
        super().__init__(skills)
        self.port = port
        self.robot_id = robot_id
        self.overhead_cam_idx = overhead_cam_idx
        self.wrist_cam_idx = wrist_cam_idx
        self.device = device
        self.warmup_s = warmup_s
        self.noise_scheduler_type = noise_scheduler_type
        self.num_inference_steps = num_inference_steps
        self.n_action_steps = n_action_steps

        self.robot = None
        self._connected = False
        self.home_qpos: np.ndarray | None = None
        self._state_keys: tuple[str, ...] = ()
        self._preprocessor: PolicyProcessorPipeline | None = None
        self._postprocessor: PolicyProcessorPipeline | None = None

        # Cache loaded policies to prevent reloading weights on every skill
        self.policies: dict[str, torch.nn.Module] = {}

    def connect(self) -> None:
        """Connect to the SO-101 bus and cameras, caching the baseline home pose."""
        if self._connected and self.robot is not None:
            return

        print(f"[LeRobotExecutor] Connecting to SO-101 on port {self.port}...")

        cameras = {
            "overhead": OpenCVCameraConfig(
                index_or_path=self.overhead_cam_idx,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
                warmup_s=self.warmup_s,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=self.wrist_cam_idx,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
                warmup_s=self.warmup_s,
            ),
        }

        cfg = SO101FollowerConfig(
            port=self.port,
            id=self.robot_id,
            cameras=cameras,
            max_relative_target=5.0,
        )

        self.robot = make_robot_from_config(cfg)
        self.robot.connect()
        self._connected = True

        # Capture initial resting pose for post-skill home resets
        time.sleep(1.0)
        initial_obs = self.robot.get_observation()
        self._state_keys = tuple(
            key for key in initial_obs if key.endswith(".pos")
        )
        if self._state_keys:
            self.home_qpos = self._extract_state(initial_obs)
            print(f"[LeRobotExecutor] Home qpos captured: {self.home_qpos.round(2)}")

        print("[LeRobotExecutor] Connected successfully.")

    def disconnect(self) -> None:
        if self._connected and self.robot is not None:
            print("[LeRobotExecutor] Disconnecting robot...")
            self.robot.disconnect()
            self._connected = False
            self.robot = None

    def _get_policy(self, skill_name: str):
        """Lazy-load and cache the policy checkpoint on the GPU."""
        if skill_name in self.policies:
            return self.policies[skill_name]

        if skill_name not in self.CHECKPOINT_PATHS:
            raise ValueError(f"Unknown skill '{skill_name}'. Must be one of {list(self.CHECKPOINT_PATHS.keys())}")

        path = self.CHECKPOINT_PATHS[skill_name]
        print(f"[LeRobotExecutor] Loading policy '{skill_name}' from {path} onto {self.device}...")

        config = DiffusionConfig.from_pretrained(path)
        config.device = self.device
        config.pretrained_path = Path(path)
        config.noise_scheduler_type = self.noise_scheduler_type
        config.num_inference_steps = self.num_inference_steps
        config.n_action_steps = self.n_action_steps
        features = {
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.wrist": PolicyFeature(
                FeatureType.VISUAL, (3, 480, 640)
            ),
            "observation.images.overhead": PolicyFeature(
                FeatureType.VISUAL, (3, 480, 640)
            ),
            "action": PolicyFeature(FeatureType.ACTION, (6,)),
        }
        policy_env = _PolicyEnvironmentConfig(
            features=features,
            features_map={key: key for key in features},
        )
        policy = make_policy(config, env_cfg=policy_env)
        self._preprocessor = PolicyProcessorPipeline.from_pretrained(
            path,
            config_filename="policy_preprocessor.json",
        )
        self._postprocessor = PolicyProcessorPipeline.from_pretrained(
            path,
            config_filename="policy_postprocessor.json",
        )

        policy.eval()
        self.policies[skill_name] = policy
        return policy

    def _extract_state(self, observation: dict[str, Any]) -> np.ndarray:
        if not self._state_keys:
            raise RuntimeError("Robot state keys have not been initialized")
        try:
            return np.asarray(
                [observation[key] for key in self._state_keys], dtype=np.float32
            )
        except KeyError as exc:
            raise KeyError(f"Missing robot state key: {exc.args[0]}") from exc

    def _action_dict(self, action: np.ndarray) -> dict[str, float]:
        if not self._state_keys:
            raise RuntimeError("Robot state keys have not been initialized")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != len(self._state_keys):
            raise ValueError(
                f"Policy returned {action.size} action values; "
                f"expected {len(self._state_keys)}"
            )
        return {
            key: float(value) for key, value in zip(self._state_keys, action)
        }

    def capture(self) -> Observation:
        """Capture current frames and return an Observation dataclass with PIL images."""
        if not self._connected or self.robot is None:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        obs = self.robot.get_observation()

        # Extract frames (handle both key naming conventions)
        overhead_raw = obs.get("observation.images.overhead", obs.get("overhead"))
        wrist_raw = obs.get("observation.images.wrist", obs.get("wrist"))

        if overhead_raw is None:
            raise KeyError("Neither 'observation.images.overhead' nor 'overhead' found")
        overhead_np = np.asarray(overhead_raw)
        if overhead_np.ndim == 3 and overhead_np.shape[0] == 3:
            overhead_np = np.transpose(overhead_np, (1, 2, 0))
        if overhead_np.dtype != np.uint8:
            overhead_np = (
                overhead_np * 255 if overhead_np.max() <= 1.0 else overhead_np
            ).astype(np.uint8)
        return Observation(np.ascontiguousarray(overhead_np), time.time())

    def _return_home(self, steps: int = 50, dt: float = 0.02) -> None:
        """Smoothly interpolate joints back to home resting position to clear the camera view."""
        if self.home_qpos is None or self.robot is None:
            return

        try:
            obs = self.robot.get_observation()
        except RuntimeError as exc:
            print(f"[LeRobotExecutor] Unable to read state for homing: {exc}")
            return
        current_qpos = self._extract_state(obs)
        home_qpos = np.asarray(
            self.home_qpos, dtype=np.float32
        )

        # Linear interpolation between current and home joint states
        for step in range(1, steps + 1):
            alpha = step / float(steps)
            target = (1.0 - alpha) * current_qpos + alpha * home_qpos
            self.robot.send_action(self._action_dict(target))
            time.sleep(dt)

    def execute_skill(self, skill_name: str, duration_s: float = 25.0) -> ExecResult:
        """Execute a single imitation learning skill rollout."""
        self.connect()
        policy = self._get_policy(skill_name)
        policy.reset()

        print(f"[LeRobotExecutor] Executing skill: {skill_name} (max duration: {duration_s}s)...")
        start_time = time.time()
        loop_dt = 1.0 / 30.0  # 30 Hz control loop

        try:
            while (time.time() - start_time) < duration_s:
                step_start = time.time()

                # 1. Read hardware observation
                raw_obs = self.robot.get_observation()

                # 2. Apply the checkpoint's saved normalization and batching pipeline.
                overhead_np = raw_obs.get("observation.images.overhead", raw_obs.get("overhead"))
                wrist_np = raw_obs.get("observation.images.wrist", raw_obs.get("wrist"))
                state_np = self._extract_state(raw_obs)

                if self._preprocessor is None or self._postprocessor is None:
                    raise RuntimeError("Policy processors were not initialized")
                raw_observation = {
                    "observation.images.overhead": np.asarray(overhead_np),
                    "observation.images.wrist": np.asarray(wrist_np),
                    "observation.state": state_np,
                }
                raw_observation = prepare_observation_for_inference(
                    raw_observation,
                    torch.device(self.device),
                )
                observation = self._preprocessor(
                    raw_observation
                )

                with torch.no_grad():
                    action_tensor = policy.select_action(observation)
                    action_tensor = self._postprocessor({"action": action_tensor})[
                        "action"
                    ]

                action_np = action_tensor.squeeze(0).cpu().numpy()

                # 4. Dispatch action to motors
                self.robot.send_action(self._action_dict(action_np))

                # Maintain control rate
                elapsed = time.time() - step_start
                if elapsed < loop_dt:
                    time.sleep(loop_dt - elapsed)

            # Return to resting pose so evaluator camera view is clear
            self._return_home()
            time.sleep(0.5)

            final_obs = self.capture()
            return ExecResult(
                status=ExecStatus.COMPLETED,
                duration_s=time.monotonic() - start_time,
            )

        except Exception as e:
            print(f"[LeRobotExecutor] Error executing {skill_name}: {e}")
            self._return_home()
            return ExecResult(
                status=ExecStatus.ERROR,
                duration_s=time.monotonic() - start_time,
                error=str(e),
            )

    def run(self, skill_name: str) -> ExecResult:
        """Execute one planned skill."""
        self._validate_skill(skill_name)
        return self.execute_skill(skill_name)

    def close(self) -> None:
        self.disconnect()