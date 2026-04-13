from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = Path("/workspace/data/oxford_iiit_pet")


def extract_breed_name(image_stem: str) -> str:
    match = re.match(r"^(.*)_\d+$", image_stem)
    return match.group(1) if match else image_stem


def load_oxford_pet_split_dataframe(
    split: str,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    split_file = dataset_root / "annotations" / f"{split}.txt"
    images_dir = dataset_root / "images"

    if not split_file.exists():
        raise FileNotFoundError(f"{split}.txt not found: {split_file}")

    columns = ["image_stem", "class_id", "species_id", "breed_id"]
    df = pd.read_csv(split_file, sep=" ", header=None, names=columns)
    df["breed_name"] = df["image_stem"].map(extract_breed_name)
    df["image_path"] = df["image_stem"].map(lambda stem: images_dir / f"{stem}.jpg")

    if not df["image_path"].map(Path.exists).all():
        raise FileNotFoundError("Some image files listed in trainval.txt are missing.")

    return df


def load_oxford_pet_trainval_dataframe(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> pd.DataFrame:
    return load_oxford_pet_split_dataframe("trainval", dataset_root=dataset_root)


def load_oxford_pet_test_dataframe(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> pd.DataFrame:
    return load_oxford_pet_split_dataframe("test", dataset_root=dataset_root)


def build_label_mapping(df: pd.DataFrame) -> dict[int, str]:
    mapping_df = df[["class_id", "breed_name"]].drop_duplicates().sort_values("class_id")
    return dict(zip(mapping_df["class_id"], mapping_df["breed_name"]))


def create_train_valid_split(
    df: pd.DataFrame,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=df["class_id"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def sample_fewshot_per_class(
    df: pd.DataFrame,
    shots_per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive.")

    sampled_groups: list[pd.DataFrame] = []
    for class_id, group in df.groupby("class_id", sort=True):
        sampled_group = group.sample(
            n=min(shots_per_class, len(group)),
            random_state=random_state + int(class_id),
        )
        sampled_groups.append(sampled_group)

    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    return sampled_df.reset_index(drop=True)


def limit_dataframe_examples(
    df: pd.DataFrame,
    max_examples: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_examples is None or max_examples >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=max_examples, random_state=random_state).reset_index(drop=True)


@dataclass(frozen=True)
class ClassificationSample:
    image: object
    label: int
    image_path: str
    image_stem: str
    breed_name: str


class OxfordIIITPetClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["class_id"]) - 1

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "image_path": str(row["image_path"]),
            "image_stem": row["image_stem"],
            "breed_name": row["breed_name"],
        }
