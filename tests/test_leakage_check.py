"""
Tests for src.data.leakage_check module.

Strategic testing approach: 3 critical tests covering the highest-risk
paths. Other functions (analyze_split, get_leakage_severity) are
covered via integration testing in notebooks/01_data_quality_eda.ipynb
against the real 4,588-image dataset.

Tests:
    1. extract_base_name handles all naming patterns correctly.
    2. check_cross_split_leakage fails loudly on invalid input.
    3. create_leakage_free_split produces zero leakage (the core
       guarantee of this module).
"""

import pytest

from src.data.leakage_check import (
    extract_base_name,
    check_cross_split_leakage,
    create_leakage_free_split,
)


# =============================================================================
# TEST 1: Base name extraction handles all filename patterns
# =============================================================================

@pytest.mark.parametrize(
    "filename,expected",
    [
        # Roboflow augmented filenames
        ("DSC05534_jpg.rf.9bfad503.jpg", "DSC05534"),
        ("IMG_001_jpg.rf.abc123def.jpg", "IMG_001"),

        # Non-Roboflow: preserve full stem
        ("store42_aisle3_shelf1.jpg", "store42_aisle3_shelf1"),
        ("simple_name.png", "simple_name"),
        ("image.jpeg", "image"),

        # Edge cases
        ("noextension", "noextension"),
        ("a.b.c.d.jpg", "a.b.c.d"),
    ],
    ids=[
        "roboflow_basic",
        "roboflow_with_underscores",
        "multi_part_name",
        "simple_with_extension",
        "basic_jpeg",
        "no_extension",
        "multiple_dots",
    ],
)
def test_extract_base_name_handles_all_patterns(filename, expected):
    """Verify base name extraction across Roboflow and non-Roboflow formats.

    This test is critical because the entire leakage detection logic
    depends on correctly grouping filenames by their source image.
    A buggy extract_base_name would cause both false positives
    (grouping unrelated images) and false negatives (missing real
    duplicates).
    """
    assert extract_base_name(filename) == expected


# =============================================================================
# TEST 2: check_cross_split_leakage fails loudly on missing keys
# =============================================================================

def test_check_cross_split_leakage_raises_on_missing_key():
    """Verify explicit error when splits dict lacks required keys.

    Common mistake: using 'val' instead of 'valid'. Silent failure
    would produce incorrect leakage results. The function must raise
    KeyError with an actionable message.
    """
    # Missing 'valid' (common mistake: 'val')
    invalid_splits = {
        "train": {"base_names": set()},
        "val": {"base_names": set()},
        "test": {"base_names": set()},
    }

    with pytest.raises(KeyError) as exc_info:
        check_cross_split_leakage(invalid_splits)

    error_message = str(exc_info.value)
    # The error message must mention which keys are missing
    assert "valid" in error_message
    # And must guide the user toward the fix
    assert "val" in error_message.lower()


def test_check_cross_split_leakage_detects_shared_base():
    """Verify that shared base names across splits are detected.

    This is the core detection logic: when the same base image
    appears in train AND valid, it must appear in train_valid leak.
    """
    splits = {
        "train": {"base_names": {"img001", "img002", "img003"}},
        "valid": {"base_names": {"img002", "img004"}},
        "test": {"base_names": {"img005", "img003"}},
    }

    result = check_cross_split_leakage(splits)

    # img002 leaks train_valid
    assert result["train_valid"] == {"img002"}
    # img003 leaks train_test
    assert result["train_test"] == {"img003"}
    # no leakage valid_test
    assert result["valid_test"] == set()
    # no image in all three
    assert result["all_three"] == set()


# =============================================================================
# TEST 3: create_leakage_free_split produces zero leakage (CORE GUARANTEE)
# =============================================================================

def test_create_leakage_free_split_produces_zero_leakage():
    """Verify the core guarantee: all augmentations of a base image
    stay in the same split.

    This is THE critical test of the module. If this fails, the entire
    remediation workflow is broken.
    """
    # Simulate a Roboflow-style dataset: 100 base images, each with
    # 3 augmentations
    all_base_to_files = {
        f"DSC{i:05d}": [
            f"DSC{i:05d}_jpg.rf.hash{j}.jpg" for j in range(3)
        ]
        for i in range(100)
    }

    result = create_leakage_free_split(
        all_base_to_files,
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Extract base names per split
    def get_bases(filenames):
        return {extract_base_name(f) for f in filenames}

    train_bases = get_bases(result["train"])
    valid_bases = get_bases(result["valid"])
    test_bases = get_bases(result["test"])

    # CORE ASSERTION: zero leakage between any pair of splits
    assert not (train_bases & valid_bases), \
        "Leakage detected: base images in both train and valid"
    assert not (train_bases & test_bases), \
        "Leakage detected: base images in both train and test"
    assert not (valid_bases & test_bases), \
        "Leakage detected: base images in both valid and test"

    # SANITY: all files are assigned (no loss)
    total_files = (
        len(result["train"])
        + len(result["valid"])
        + len(result["test"])
    )
    expected_total = sum(len(v) for v in all_base_to_files.values())
    assert total_files == expected_total, \
        f"Files lost: expected {expected_total}, got {total_files}"


def test_create_leakage_free_split_raises_on_invalid_ratios():
    """Verify ratio validation prevents silent errors.

    A common mistake is passing ratios that don't sum to 1.0.
    The function must raise ValueError with explicit message.
    """
    all_base_to_files = {
        "img001": ["img001_v1.jpg"],
        "img002": ["img002_v1.jpg"],
        "img003": ["img003_v1.jpg"],
    }

    with pytest.raises(ValueError) as exc_info:
        create_leakage_free_split(
            all_base_to_files,
            train_ratio=0.80,   # 0.80 + 0.15 + 0.15 = 1.10
            valid_ratio=0.15,
            test_ratio=0.15,
        )

    assert "1.0" in str(exc_info.value)
