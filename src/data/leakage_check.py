"""
Data Quality Validation: Leakage Detection and Remediation Module
==================================================================

This module detects and remediates data leakage in YOLO11 datasets
where the same base image (before augmentations) may appear across
multiple splits (train/valid/test).

Roboflow and similar tools apply augmentations per-image and generate
new filenames with hash suffixes (e.g., 'DSC05534_jpg.rf.9bfad503...jpg').
If these augmentations leak across splits, the model effectively
evaluates on data it has already seen, producing artificially inflated
validation metrics.

Additionally, this module validates YOLO-format datasets by ensuring
each image has a corresponding label file (orphan detection).

This module provides:
    - Base name extraction from augmented filenames
    - Per-split analysis with orphan detection
    - Cross-split leakage detection with explicit validation
    - Severity classification (CLEAN/MINOR/MODERATE/SEVERE)
    - Structured reporting for CSV export
    - Group-aware split regeneration with ratio transparency

Usage:
    from src.data.leakage_check import (
        analyze_split,
        check_cross_split_leakage,
        create_leakage_free_split,
    )

    train_stats = analyze_split(Path("data/raw/train"))
    leaks = check_cross_split_leakage(splits_data)
    new_splits = create_leakage_free_split(all_base_to_files)
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Module-level logger following library best practices
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Filename pattern constants
ROBOFLOW_AUGMENTATION_SEPARATOR = "_jpg.rf."
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_EXTENSION = ".txt"


def extract_base_name(filename: str) -> str:
    """Extract the base image name from an augmented filename.

    Roboflow generates filenames in the format:
        <BASE_NAME>_jpg.rf.<HASH>.jpg

    For Roboflow-augmented files, returns the BASE_NAME portion.
    For non-Roboflow files, returns the filename stem (without
    extension) to preserve multi-part names like 'store42_aisle3'.

    Args:
        filename: The full filename including extension.

    Returns:
        The base name identifying the original source image.

    Examples:
        >>> extract_base_name("DSC05534_jpg.rf.9bfad503.jpg")
        'DSC05534'
        >>> extract_base_name("store42_aisle3_shelf1.jpg")
        'store42_aisle3_shelf1'
        >>> extract_base_name("image.jpg")
        'image'
    """
    if ROBOFLOW_AUGMENTATION_SEPARATOR in filename:
        return filename.split(ROBOFLOW_AUGMENTATION_SEPARATOR)[0]
    # Preserve full stem for non-Roboflow naming conventions
    return Path(filename).stem


def analyze_split(split_path: Path) -> dict[str, Any]:
    """Analyze a single split for augmentation patterns and orphan images.

    Counts total images, unique base names, augmentation factor, and
    identifies orphan images (images without corresponding label files).
    YOLO silently discards orphan images during training, which distorts
    reported dataset sizes.

    Args:
        split_path: Path to the split directory. Must contain 'images/'
            and ideally 'labels/' subdirectories.

    Returns:
        A dictionary with keys:
            - total_images: Total number of image files.
            - unique_bases: Number of unique base images.
            - base_names: Set of all base names found.
            - augmentation_factor: Ratio of total/unique.
            - base_to_files: Mapping base_name -> list of filenames.
            - orphan_images: List of images without matching .txt labels.
            - orphan_count: Number of orphan images.
            - valid_images: Images with matching labels (for training).
    """
    images_path = split_path / "images"
    labels_path = split_path / "labels"

    if not images_path.exists():
        return _empty_split_result()

    base_to_files: dict[str, list[str]] = defaultdict(list)
    orphan_images: list[str] = []
    valid_images: list[str] = []

    labels_exist = labels_path.exists()

    for img_file in images_path.iterdir():
        if img_file.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        base = extract_base_name(img_file.name)
        base_to_files[base].append(img_file.name)

        # Check for corresponding label file
        if labels_exist:
            label_file = labels_path / (img_file.stem + LABEL_EXTENSION)
            if label_file.exists():
                valid_images.append(img_file.name)
            else:
                orphan_images.append(img_file.name)

    total = sum(len(files) for files in base_to_files.values())
    unique = len(base_to_files)
    aug_factor = total / unique if unique > 0 else 0.0

    # Log warning if orphans detected
    if orphan_images:
        logger.warning(
            f"Found {len(orphan_images)} orphan images in "
            f"'{split_path.name}' (no corresponding .txt label). "
            f"YOLO will silently skip these during training."
        )

    return {
        "total_images": total,
        "unique_bases": unique,
        "base_names": set(base_to_files.keys()),
        "augmentation_factor": aug_factor,
        "base_to_files": dict(base_to_files),
        "orphan_images": orphan_images,
        "orphan_count": len(orphan_images),
        "valid_images": valid_images,
    }


def _empty_split_result() -> dict[str, Any]:
    """Return empty result structure for missing split directories."""
    return {
        "total_images": 0,
        "unique_bases": 0,
        "base_names": set(),
        "augmentation_factor": 0.0,
        "base_to_files": {},
        "orphan_images": [],
        "orphan_count": 0,
        "valid_images": [],
    }


def check_cross_split_leakage(
    splits_data: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """Identify base names that appear across multiple splits.

    Leakage occurs when the same source image (before augmentation)
    ends up in more than one split. The model effectively evaluates
    on data it has already seen during training, inflating metrics.

    Args:
        splits_data: Dictionary mapping split names to analysis results.
            Must contain keys 'train', 'valid', 'test', each with a
            'base_names' set.

    Returns:
        A dictionary with keys:
            - train_valid: Base names in both train and valid.
            - train_test: Base names in both train and test.
            - valid_test: Base names in both valid and test.
            - all_three: Base names in all three splits.

    Raises:
        KeyError: If splits_data is missing any of the required keys
            'train', 'valid', or 'test'. Common mistake: using 'val'
            instead of 'valid'.
    """
    required_splits = {"train", "valid", "test"}
    missing = required_splits - set(splits_data.keys())
    if missing:
        raise KeyError(
            f"Missing required split(s): {sorted(missing)}. "
            f"Got splits: {sorted(splits_data.keys())}. "
            f"Note: this function expects 'valid' (not 'val')."
        )

    train_bases = splits_data["train"]["base_names"]
    valid_bases = splits_data["valid"]["base_names"]
    test_bases = splits_data["test"]["base_names"]

    return {
        "train_valid": train_bases & valid_bases,
        "train_test": train_bases & test_bases,
        "valid_test": valid_bases & test_bases,
        "all_three": train_bases & valid_bases & test_bases,
    }


def generate_leakage_report(leaks: dict[str, set[str]]) -> pd.DataFrame:
    """Generate a structured DataFrame of all leakage cases.

    Args:
        leaks: Output from check_cross_split_leakage().

    Returns:
        DataFrame with columns ['base_name', 'leak_type'].
        One row per leaking base image.
    """
    records = []
    for leak_type, base_names in leaks.items():
        for base in base_names:
            records.append(
                {
                    "base_name": base,
                    "leak_type": leak_type,
                }
            )

    return pd.DataFrame(records)


def build_split_summary(
    splits_data: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Create a summary DataFrame across all splits for visualization.

    Args:
        splits_data: Dictionary from analyze_split() per split.

    Returns:
        DataFrame with split statistics including orphan counts.
    """
    total_all = sum(s["total_images"] for s in splits_data.values())

    records = []
    for split_name, stats in splits_data.items():
        records.append(
            {
                "split": split_name,
                "total_images": stats["total_images"],
                "unique_bases": stats["unique_bases"],
                "augmentation_factor": round(stats["augmentation_factor"], 2),
                "orphan_count": stats["orphan_count"],
                "valid_images": len(stats["valid_images"]),
                "percentage": (
                    round(100 * stats["total_images"] / total_all, 1)
                    if total_all > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(records)


def get_leakage_severity(
    leaks: dict[str, set[str]],
    splits_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Quantify the severity of detected leakage with logged warnings.

    Severity levels:
        - CLEAN: No leakage detected.
        - MINOR: < 1% of unique images leak.
        - MODERATE: 1-5% leak (requires attention).
        - SEVERE: > 5% leak (invalidates metrics, must fix).

    Args:
        leaks: Output from check_cross_split_leakage().
        splits_data: Output from analyze_split() per split.

    Returns:
        Dictionary with total_leaked_bases, leak_rate_pct, and status.
    """
    all_leaked = leaks["train_valid"] | leaks["train_test"] | leaks["valid_test"]
    total_leaked = len(all_leaked)

    total_unique = len(
        splits_data["train"]["base_names"]
        | splits_data["valid"]["base_names"]
        | splits_data["test"]["base_names"]
    )

    leak_rate = 100 * total_leaked / total_unique if total_unique > 0 else 0.0

    if total_leaked == 0:
        status = "CLEAN"
    elif leak_rate < 1.0:
        status = "MINOR"
    elif leak_rate < 5.0:
        status = "MODERATE"
    else:
        status = "SEVERE"

    # Senior-style pedagogical logging
    if status == "SEVERE":
        logger.error(
            f"SEVERE data leakage detected: {leak_rate:.1f}% of unique "
            f"base images leak across splits. Training with this split "
            f"will produce artificially inflated validation metrics. "
            f"Apply create_leakage_free_split() before training."
        )
    elif status == "MODERATE":
        logger.warning(
            f"MODERATE data leakage detected: {leak_rate:.1f}% of "
            f"unique base images leak across splits. Consider "
            f"regenerating splits with create_leakage_free_split()."
        )
    elif status == "MINOR":
        logger.info(
            f"MINOR leakage detected: {leak_rate:.1f}%. Impact on "
            f"metrics likely negligible but documented for transparency."
        )

    return {
        "total_leaked_bases": total_leaked,
        "total_unique_bases": total_unique,
        "leak_rate_pct": round(leak_rate, 2),
        "status": status,
    }


def create_leakage_free_split(
    all_base_to_files: dict[str, list[str]],
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Regenerate train/valid/test splits grouped by base image.

    Uses GroupShuffleSplit to ensure ALL augmented versions of a
    source image remain in the SAME split, eliminating cross-split
    leakage. This is the canonical scikit-learn solution for
    group-aware splitting.

    Note: GroupShuffleSplit distributes GROUPS proportionally, not
    files. When augmentation counts per group are uneven, the actual
    file ratios may drift from the requested ratios. This function
    logs both requested and actual ratios for transparency.

    Args:
        all_base_to_files: Combined mapping from all splits.
            Key: base_name, Value: list of augmented filenames.
        train_ratio: Proportion for training set (default 0.70).
        valid_ratio: Proportion for validation set (default 0.15).
        test_ratio: Proportion for test set (default 0.15).
        random_state: Seed for reproducibility (default 42).

    Returns:
        Dictionary with keys 'train', 'valid', 'test', each containing
        the list of filenames assigned to that split.

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance), or
            if fewer than 3 unique base images are provided.
    """
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={train_ratio}, valid={valid_ratio}, "
            f"test={test_ratio})"
        )

    base_names = list(all_base_to_files.keys())
    n_bases = len(base_names)

    if n_bases < 3:
        raise ValueError(
            f"Need at least 3 unique base images for a 3-way split, " f"got {n_bases}."
        )

    # Create groups array: each file belongs to a group (base_name)
    files = []
    groups = []
    for base_name, filenames in all_base_to_files.items():
        for fname in filenames:
            files.append(fname)
            groups.append(base_name)

    files_array = np.array(files)
    groups_array = np.array(groups)
    indices = np.arange(len(files))

    # First split: train vs (valid + test)
    test_valid_ratio = valid_ratio + test_ratio
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_valid_ratio,
        random_state=random_state,
    )
    train_idx, temp_idx = next(gss1.split(indices, groups=groups_array))

    # Second split: valid vs test from the temp set
    temp_groups = groups_array[temp_idx]
    temp_indices = np.arange(len(temp_idx))
    test_portion = test_ratio / test_valid_ratio

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_portion,
        random_state=random_state,
    )
    valid_rel_idx, test_rel_idx = next(gss2.split(temp_indices, groups=temp_groups))

    # Map back to original indices
    valid_idx = temp_idx[valid_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    result = {
        "train": files_array[train_idx].tolist(),
        "valid": files_array[valid_idx].tolist(),
        "test": files_array[test_idx].tolist(),
    }

    # Verify no leakage in new split
    train_groups = set(groups_array[train_idx])
    valid_groups = set(groups_array[valid_idx])
    test_groups = set(groups_array[test_idx])

    leakage_check = (
        bool(train_groups & valid_groups)
        or bool(train_groups & test_groups)
        or bool(valid_groups & test_groups)
    )

    if leakage_check:
        logger.error("create_leakage_free_split produced leakage. This is a bug.")
    else:
        logger.info(
            f"Clean split generated: {len(train_idx)} train, "
            f"{len(valid_idx)} valid, {len(test_idx)} test files. "
            f"Zero cross-split leakage verified."
        )

    # Log actual vs requested split ratios for transparency
    # (GroupShuffleSplit splits by groups, not files; ratios may drift
    # when augmentation counts per group are uneven)
    total_files = len(files)
    actual_train_pct = round(100 * len(train_idx) / total_files, 1)
    actual_valid_pct = round(100 * len(valid_idx) / total_files, 1)
    actual_test_pct = round(100 * len(test_idx) / total_files, 1)

    logger.info(
        f"Split ratios — requested: "
        f"train={train_ratio*100:.1f}% / "
        f"valid={valid_ratio*100:.1f}% / "
        f"test={test_ratio*100:.1f}% | "
        f"actual: "
        f"train={actual_train_pct}% / "
        f"valid={actual_valid_pct}% / "
        f"test={actual_test_pct}%"
    )

    return result
