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
    task: str
    annotation_format: str
    pid: int
    images: list[ImageRecord] = field(default_factory=list)
    clothes_id: int | None = None
    modality: str = "rgb"
    group_id: str = ""

    def __post_init__(self) -> None:
        self.dataset = self.dataset.strip().lower()
        self.task = self.task.strip().lower()
        self.annotation_format = self.annotation_format.strip().lower()
        self.modality = self.modality.strip().lower()
        if self.task not in {
            "traditional",
            "anytime",
            "clothes-changing",
            "visible-infrared",
        }:
            raise ValueError(f"Unsupported ReID task: {self.task}")
        if self.annotation_format not in {"dense", "diverse"}:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
        if self.modality not in {"rgb", "ir"}:
            raise ValueError(f"Unsupported modality: {self.modality}")
        if self.annotation_format == "diverse" and self.clothes_id is None:
            raise ValueError("Diverse annotation groups require clothes_id")
        if not self.group_id:
            if self.annotation_format == "dense":
                self.group_id = f"{self.dataset}_pid{self.pid:04d}"
            else:
                self.group_id = (
                    f"{self.dataset}_pid{self.pid:04d}_"
                    f"clothes{int(self.clothes_id):03d}_{self.modality}"
                )

    @property
    def resume_key(self) -> tuple[int | str, ...]:
        if self.annotation_format == "dense":
            return (self.pid,)
        return (self.pid, int(self.clothes_id), self.modality)
