"""
Verify dataset safety rules without running training.

  python check_dataset_safety.py
"""

from fedsm.safety import (
    CAROTID_IMAGE_DIR,
    CAROTID_MASK_DIR,
    CAROTID_MODEL,
    CAROTID_HISTORY,
    VENOUS_IMAGE_DIR,
    VENOUS_MASK_DIR,
    VENOUS_MODEL,
    VENOUS_HISTORY,
    assert_training_config,
    assert_safe_output_dir,
    DatasetSafetyError,
)


def expect_ok(fn, label):
    try:
        fn()
        print(f"  OK: {label}")
    except DatasetSafetyError as e:
        print(f"  FAIL (should pass): {label}\n    {e}")


def expect_fail(fn, label):
    try:
        fn()
        print(f"  FAIL (should refuse): {label}")
    except DatasetSafetyError:
        print(f"  OK refused: {label}")


if __name__ == "__main__":
    print("=== Carotid training config ===")
    expect_ok(
        lambda: assert_training_config(
            CAROTID_IMAGE_DIR, CAROTID_MASK_DIR, CAROTID_MODEL, CAROTID_HISTORY, "carotid"
        ),
        "carotid paths + carotid model",
    )
    expect_fail(
        lambda: assert_training_config(
            VENOUS_IMAGE_DIR, VENOUS_MASK_DIR, CAROTID_MODEL, CAROTID_HISTORY, "carotid"
        ),
        "carotid training on venous folders",
    )
    expect_fail(
        lambda: assert_training_config(
            CAROTID_IMAGE_DIR, CAROTID_MASK_DIR, VENOUS_MODEL, CAROTID_HISTORY, "carotid"
        ),
        "carotid training overwriting venous model",
    )

    print("\n=== Venous training config ===")
    expect_ok(
        lambda: assert_training_config(
            VENOUS_IMAGE_DIR, VENOUS_MASK_DIR, VENOUS_MODEL, VENOUS_HISTORY, "venous"
        ),
        "venous paths + venous model",
    )
    expect_fail(
        lambda: assert_training_config(
            CAROTID_IMAGE_DIR, CAROTID_MASK_DIR, VENOUS_MODEL, VENOUS_HISTORY, "venous"
        ),
        "venous training on carotid folders",
    )

    print("\n=== Protected folder writes ===")
    expect_fail(
        lambda: assert_safe_output_dir("images", confirm_overwrite_venous=False),
        "extract to images/ without confirm",
    )
    expect_ok(
        lambda: assert_safe_output_dir("images", confirm_overwrite_venous=True),
        "extract to images/ with confirm",
    )
    expect_ok(
        lambda: assert_safe_output_dir("carotid_output_test", confirm_overwrite_venous=False),
        "write to non-protected folder",
    )

    print("\nAll safety checks completed.")
