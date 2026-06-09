import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track


class FixedLabelEncoder:
    """Small encoder object compatible with the Synesis downstream script."""

    def __init__(self, classes):
        self.classes_ = np.array(classes)


class NSynth(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "/content/drive/MyDrive/datasets/nsynth_reduced_5k",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "wav",
        item_format: str = "feature",
        itemization: bool = True,
        seed: int = 42,
        label: str = "instrument_family",
        transform=None,
    ) -> None:
        self.tasks = ["instrument_family_classification", "qualities_classification"]
        self.fvs = ["instrument_family", "qualities"]

        if label not in self.fvs:
            raise ValueError(f"Invalid NSynth label: {label}. Options: {self.fvs}")

        if split not in [None, "train", "validation", "valid", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Options: None, 'train', 'validation', 'valid', 'test'"
            )

        self.feature = feature
        self.root = Path(root)
        self.split = "validation" if split == "valid" else split
        self.label = label
        self.item_format = item_format
        self.itemization = itemization
        self.audio_format = audio_format

        if not feature_config:
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        if download:
            self._download()

        if self.label == "instrument_family":
            self.label_encoder = FixedLabelEncoder(list(range(11)))
        else:
            self.label_encoder = FixedLabelEncoder(
                [
                    "bright",
                    "dark",
                    "distortion",
                    "fast_decay",
                    "long_release",
                    "multiphonic",
                    "nonlinear_env",
                    "percussive",
                    "reverb",
                    "tempo-synced",
                ]
            )

        self._load_metadata()

    def _download(self) -> None:
        raise NotImplementedError(
            "NSynth download is not implemented here. Place the dataset under the expected root "
            "or pass root=/path/to/nsynth."
        )

    def _candidate_split_dirs(self):
        if self.split is None:
            return [
                self.root / "train",
                self.root / "validation",
                self.root / "test",
                self.root / "nsynth-train",
                self.root / "nsynth-valid",
                self.root / "nsynth-test",
            ]

        mapping = {
            "train": ["train", "nsynth-train", "nsynth_train"],
            "validation": ["validation", "valid", "nsynth-valid", "nsynth-validation", "nsynth_valid"],
            "test": ["test", "nsynth-test", "nsynth_test"],
        }
        return [self.root / name for name in mapping[self.split]]

    def _existing_split_dirs(self):
        split_dirs = [p for p in self._candidate_split_dirs() if (p / "examples.json").exists()]
        if not split_dirs:
            candidates = "\\n".join(str(p / "examples.json") for p in self._candidate_split_dirs())
            raise FileNotFoundError(
                "Could not find NSynth examples.json for the requested split. Tried:\\n"
                + candidates
            )
        return split_dirs

    def _audio_path_for_note(self, split_root: Path, note_id: str) -> Path:
        direct = split_root / f"{note_id}.{self.audio_format}"
        nested = split_root / "audio" / f"{note_id}.{self.audio_format}"
        if direct.exists():
            return direct
        return nested

    def _feature_path_for_note(self, split_root: Path, note_id: str) -> Path:
        return split_root / self.feature / f"{note_id}.pt"

    def _qualities_to_vector(self, qualities, note_id: str):
        if isinstance(qualities, dict):
            return [float(qualities[name]) for name in self.label_encoder.classes_]

        if len(qualities) != len(self.label_encoder.classes_):
            raise ValueError(
                f"Expected 10 NSynth qualities, got {len(qualities)} for note {note_id}"
            )

        return [float(q) for q in qualities]

    def _load_metadata(self) -> Tuple[list, torch.Tensor]:
        raw_data_paths = []
        feature_paths = []
        labels = []

        for split_root in self._existing_split_dirs():
            metadata_path = split_root / "examples.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            for note_id, example in sorted(metadata.items()):
                raw_data_paths.append(self._audio_path_for_note(split_root, note_id))
                feature_paths.append(self._feature_path_for_note(split_root, note_id))

                if self.label == "instrument_family":
                    labels.append(int(example["instrument_family"]))
                elif self.label == "qualities":
                    labels.append(self._qualities_to_vector(example["qualities"], note_id))

        self.raw_data_paths = raw_data_paths
        self.feature_paths = feature_paths

        if self.label == "instrument_family":
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float)

        self.paths = self.raw_data_paths if self.item_format == "raw" else self.feature_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = self.raw_data_paths[idx] if self.item_format == "raw" else self.feature_paths[idx]
        label = self.labels[idx]

        item_len_sec = self.feature_config.get("item_len_sec", None)

        track = load_track(
            path=path,
            item_format=self.item_format,
            itemization=self.itemization,
            item_len_sec=item_len_sec,
            sample_rate=self.feature_config["sample_rate"],
        )

        if self.item_format == "feature":
            track = track.reshape(-1)

        return track, label


Nsynth = NSynth
