"""Shared classification utilities."""
from .data import (
    DEFAULT_DATASET_ROOT,
    OxfordIIITPetClassificationDataset,
    build_label_mapping,
    create_train_valid_split,
    load_oxford_pet_split_dataframe,
    load_oxford_pet_test_dataframe,
    load_oxford_pet_trainval_dataframe,
    limit_dataframe_examples,
    sample_fewshot_per_class,
)
from .evaluation import (
    build_classification_report,
    build_confusion_matrix,
    collect_predictions,
)
from .hf import build_hf_processor_collate_fn
from .preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OPENCLIP_MEAN,
    OPENCLIP_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
    build_openclip_eval_transform,
    build_siglip_eval_transform,
    build_supervised_eval_transform,
    build_supervised_train_transform,
)
from .training import evaluate_one_epoch, fit_classifier, train_one_epoch
from .tuning import expand_grid, load_best_trial, rank_tuning_results, save_tuning_artifacts

__all__ = [
    "DEFAULT_DATASET_ROOT",
    "OxfordIIITPetClassificationDataset",
    "build_label_mapping",
    "create_train_valid_split",
    "load_oxford_pet_split_dataframe",
    "load_oxford_pet_test_dataframe",
    "load_oxford_pet_trainval_dataframe",
    "limit_dataframe_examples",
    "sample_fewshot_per_class",
    "build_classification_report",
    "build_confusion_matrix",
    "collect_predictions",
    "build_hf_processor_collate_fn",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "OPENCLIP_MEAN",
    "OPENCLIP_STD",
    "SIGLIP_MEAN",
    "SIGLIP_STD",
    "build_openclip_eval_transform",
    "build_siglip_eval_transform",
    "build_supervised_eval_transform",
    "build_supervised_train_transform",
    "evaluate_one_epoch",
    "fit_classifier",
    "train_one_epoch",
    "expand_grid",
    "load_best_trial",
    "rank_tuning_results",
    "save_tuning_artifacts",
]
