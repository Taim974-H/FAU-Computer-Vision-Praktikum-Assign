from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ReIdMode(Enum):
    IDENT = "ident"
    """The re-identification mode via classification."""
    CLUSTER = "cluster"
    """The re-identification mode via clustering."""


def enum_choices(enum_type: type[Enum]):
    return list(enum_type) + [e.value for e in enum_type]


@dataclass
class Config:
    project_dir = Path(__file__).parents[2]
    """The project directory path."""

    data_dir = Path(project_dir.joinpath("data"))
    """The data directory path."""

    train_data = data_dir.joinpath("training_data")
    test_data = data_dir.joinpath("test_data")

    # specific files
    resnet50 = data_dir.joinpath("resnet50_128.onnx")
    cluster_gallery = data_dir.joinpath("clustering_gallery.pkl")
    rec_gallery = data_dir.joinpath("recognition_gallery.pkl")

    eval_train_data = data_dir.joinpath("evaluation_training_data.pkl")
    eval_test_data = data_dir.joinpath("evaluation_test_data.pkl")

    chal_val_data = data_dir.joinpath("challenge_validation_data.csv")
    chal_test_data = data_dir.joinpath("challenge_test_data.csv")
