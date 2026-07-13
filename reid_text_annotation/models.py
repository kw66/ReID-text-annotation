from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    camera_id: int | None = None
    split: str = ""
    frame_index: int = 0


@dataclass
class AnnotationGroup:
    dataset: str
    protocol: str
    pid: int
    images: list[ImageRecord] = field(default_factory=list)
    clothes_id: int | None = None
    modality: str = "rgb"
    group_id: str = ""

    def __post_init__(self) -> None:
        self.dataset = self.dataset.strip().lower()
        self.protocol = self.protocol.strip().lower()
        self.modality = self.modality.strip().lower()
        if self.protocol not in {"public-rgb", "rgb-ir"}:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
        if self.modality not in {"rgb", "ir"}:
            raise ValueError(f"Unsupported modality: {self.modality}")
        if self.protocol == "rgb-ir" and self.clothes_id is None:
            raise ValueError("rgb-ir groups require clothes_id")
        if not self.group_id:
            if self.protocol == "public-rgb":
                self.group_id = f"{self.dataset}_pid{self.pid:04d}"
            else:
                self.group_id = (
                    f"{self.dataset}_pid{self.pid:04d}_"
                    f"clothes{int(self.clothes_id):03d}_{self.modality}"
                )

    @property
    def resume_key(self) -> tuple[int | str, ...]:
        if self.protocol == "public-rgb":
            return (self.pid,)
        return (self.pid, int(self.clothes_id), self.modality)
