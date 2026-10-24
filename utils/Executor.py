from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .Skills import Skills


class ExecStatus(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class ExecResult:
    status: ExecStatus
    duration_s: float
    joint_log_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Observation:
    overhead: np.ndarray  # H x W x 3 RGB
    timestamp: float  # Unix timestamp in seconds


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
        """Prepare the backend for execution"""

    @abstractmethod
    def run(self, skill_name: str) -> ExecResult:
        """
        Validate and execute one supported skill by name.

        Return:
        1. COMPLETED OR
        2. TIMEOUT OR
        3. ERROR with error meesage
        """

    @abstractmethod
    def capture(self) -> Observation:
        """
        Return the current overhead images and timestamp
        """

    @abstractmethod
    def close(self) -> None:
        """
        Release the connection
        """


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
            raise TypeError(
                "after_images must be a list or tuple of image paths or RGB arrays"
            )

        self.request = request
        self._images = [self._load_image(before_image)]
        self._images.extend(self._load_image(image) for image in after_images)
        self._index = 0
        self._closed = False
        self._batch_open = False  # True after a run, until the next capture.
        self._observed_at = time.time()
        self.last_skill: str | None = None

    @property
    def remaining_steps(self) -> int:
        return len(self._images) - self._index - 1

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(Path(image).expanduser()) as source:
                image = np.array(source.convert("RGB"))
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Images must be file paths or uint8 RGB arrays")
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
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
            if not self._batch_open and self.remaining_steps == 0:
                raise RuntimeError(
                    "No feedback images remaining: playback is exhausted"
                )
        except (ValueError, RuntimeError) as exc:
            return ExecResult(
                ExecStatus.ERROR, time.monotonic() - started, error=str(exc)
            )

        if not self._batch_open:
            self._index += 1
            self._observed_at = time.time()
            self._batch_open = True
        self.last_skill = skill_name
        return ExecResult(ExecStatus.COMPLETED, time.monotonic() - started)

    def capture(self) -> Observation:
        if self._closed:
            raise RuntimeError("Simulator is closed; call connect() to reopen it")
        self._batch_open = False
        return Observation(
            overhead=self._images[self._index].copy(),
            timestamp=self._observed_at,
        )

    def close(self) -> None:
        self._closed = True


# CHANGE_ME: REPLACE THIS WITH REAL ROBOT EXECUTOR CLASS
class LeRobotExecutor(Executor):
    # This is a placeholder for the actual executor that would interface with the LeRobot hardware.
    # Need to implement at least the connect, run, capture and close methods.
    def __init__(self, skills: Skills):
        self._skills = skills
        pass
