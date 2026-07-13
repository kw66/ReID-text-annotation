from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reid_text_annotation.api import parse_json_content
from reid_text_annotation.discovery import discover_atustc, discover_public_rgb, load_manifest
from reid_text_annotation.models import ImageRecord
from reid_text_annotation.models import AnnotationGroup
from reid_text_annotation.pipeline import AnnotationPipeline
from reid_text_annotation.selection import image_data_url, select_images


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (80, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


class DiscoveryTests(unittest.TestCase):
    def test_market1501_groups_splits_and_skips_junk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "bounding_box_train" / "0001_c1_0001.jpg", (255, 0, 0))
            _write_image(root / "query" / "0001_c2_0002.jpg", (0, 255, 0))
            _write_image(root / "bounding_box_test" / "0002_c3_0003.jpg", (0, 0, 255))
            _write_image(root / "bounding_box_test" / "-1_c4_0004.jpg", (0, 0, 0))
            groups = discover_public_rgb(root, "market1501")
            self.assertEqual([group.pid for group in groups], [1, 2])
            self.assertEqual(len(groups[0].images), 2)
            self.assertEqual({item.camera_id for item in groups[0].images}, {1, 2})

    def test_atustc_groups_by_clothes_and_modality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "p0001-d001-c002" / "cam1-f1-000001.jpg", (50, 50, 50))
            _write_image(root / "p0001-d001-c002" / "cam9-f1-000002.jpg", (90, 90, 90))
            groups = discover_atustc(root)
            self.assertEqual(len(groups), 2)
            self.assertEqual({group.modality for group in groups}, {"rgb", "ir"})
            self.assertTrue(all(group.clothes_id == 2 for group in groups))

    def test_manifest_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "sample.jpg"
            _write_image(image_path, (10, 20, 30))
            manifest = root / "groups.jsonl"
            manifest.write_text(
                json.dumps({"dataset": "custom", "pid": 7, "images": ["images/sample.jpg"]}) + "\n",
                encoding="utf-8",
            )
            groups = load_manifest(manifest, "public-rgb")
            self.assertEqual(groups[0].images[0].path, image_path.resolve())


class SelectionAndApiTests(unittest.TestCase):
    def test_selection_prefers_camera_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / f"sample_{index}.jpg" for index in range(3)]
            for index, path in enumerate(paths):
                _write_image(path, (index * 80, 30, 200 - index * 50))
            records = [
                ImageRecord(paths[0], camera_id=1),
                ImageRecord(paths[1], camera_id=1),
                ImageRecord(paths[2], camera_id=2),
            ]
            selected = select_images(records, 2)
            self.assertEqual({item.camera_id for item in selected}, {1, 2})
            self.assertTrue(image_data_url(selected[0]).startswith("data:image/jpeg;base64,"))

    def test_json_fence_is_accepted(self) -> None:
        payload = parse_json_content('```json\n{"dense_en":"example"}\n```')
        self.assertEqual(payload["dense_en"], "example")

    def test_rgb_ir_pipeline_emits_release_schema(self) -> None:
        class FakeApi:
            calls = 0

            def complete_json(self, **_kwargs):
                self.calls += 1
                if self.calls <= 4:
                    return {"cues": ["visible cue"], "uncertain_cues": []}
                styles = [
                    "dense_full", "person_core", "outfit_core",
                    "distinctive_mix", "retrieval_balanced",
                    "random_a", "random_b", "random_c", "random_d", "random_e",
                ]
                return {
                    "dense_en": "A compact evidence-grounded description.",
                    "captions": [
                        {"style": style, "en": f"Caption for {style}."}
                        for style in styles
                    ],
                }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.jpg"
            _write_image(path, (30, 60, 90))
            group = AnnotationGroup(
                dataset="custom",
                protocol="rgb-ir",
                pid=1,
                clothes_id=2,
                modality="ir",
                images=[ImageRecord(path=path, camera_id=9)],
            )
            result = AnnotationPipeline(api=FakeApi(), max_images=1).annotate_group(group)
            self.assertEqual(set(result), {"pid", "clothes_id", "modality", "dense_en", "captions"})
            self.assertEqual(len(result["captions"]), 10)
            self.assertEqual(result["captions"][0]["style"], "dense_full")


if __name__ == "__main__":
    unittest.main()
