#!/usr/bin/env python3
"""
Benchmark script to compare Docker output against reference segmentation mask.

This script compares the segmentation mask from the Docker container
(example_test/lesion_msk_docker.nii.gz) against the reference/local prediction
(example_test/lesion_msk.nii.gz) using various metrics.

Usage:
    python src/benchmark.py
    python -m src.benchmark
"""
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from medpy.metric import binary
from scipy.spatial.distance import directed_hausdorff

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_TEST_DIR = PROJECT_ROOT / "example_test"
REFERENCE_MASK = EXAMPLE_TEST_DIR / "lesion_msk.nii.gz"
DOCKER_MASK = EXAMPLE_TEST_DIR / "lesion_msk_docker.nii.gz"


def load_mask(filepath):
    """Load a NIfTI mask file and return as binary array."""
    if not filepath.exists():
        return None
    nii = nib.load(str(filepath))
    data = nii.get_fdata()
    # Ensure binary mask (0 or 1)
    return (data > 0.5).astype(np.uint8)


def compute_dice_coefficient(mask1, mask2):
    """Compute Dice coefficient between two binary masks."""
    if mask1 is None or mask2 is None:
        return None

    # Ensure same shape
    if mask1.shape != mask2.shape:
        return None

    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()

    if union == 0:
        return 1.0  # Both masks are empty, perfect match

    dice = 2.0 * intersection / union
    return float(dice)


def compute_hausdorff_distance(mask1, mask2, percentile=95):
    """
    Compute Hausdorff distance (percentile) between two binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask
        percentile: Percentile to use (default 95)

    Returns:
        Hausdorff distance at specified percentile
    """
    if mask1 is None or mask2 is None:
        return None

    if mask1.shape != mask2.shape:
        return None

    # Get coordinates of non-zero voxels
    coords1 = np.argwhere(mask1 > 0)
    coords2 = np.argwhere(mask2 > 0)

    if len(coords1) == 0 and len(coords2) == 0:
        return 0.0  # Both masks are empty

    if len(coords1) == 0 or len(coords2) == 0:
        # One mask is empty, return maximum possible distance
        return float(np.sqrt(sum(d**2 for d in mask1.shape)))

    # Compute directed Hausdorff distances
    hausdorff_1_to_2 = directed_hausdorff(coords1, coords2)[0]
    hausdorff_2_to_1 = directed_hausdorff(coords2, coords1)[0]

    # Symmetric Hausdorff distance
    hausdorff = max(hausdorff_1_to_2, hausdorff_2_to_1)

    # For percentile, we'd need to compute all distances, but for simplicity
    # we'll use the maximum. For true 95th percentile, use MedPy's function
    try:
        hausdorff_95 = binary.hd95(mask1, mask2)
        return float(hausdorff_95)
    except Exception:
        # Fallback to maximum Hausdorff if MedPy fails
        return float(hausdorff)


def compute_sensitivity_specificity(mask1, mask2):
    """
    Compute sensitivity (recall) and specificity between two binary masks.

    Args:
        mask1: Reference mask (ground truth)
        mask2: Predicted mask

    Returns:
        Tuple of (sensitivity, specificity)
    """
    if mask1 is None or mask2 is None:
        return None, None

    if mask1.shape != mask2.shape:
        return None, None

    # True positives, false positives, false negatives, true negatives
    tp = np.logical_and(mask1 == 1, mask2 == 1).sum()
    fp = np.logical_and(mask1 == 0, mask2 == 1).sum()
    fn = np.logical_and(mask1 == 1, mask2 == 0).sum()
    tn = np.logical_and(mask1 == 0, mask2 == 0).sum()

    # Sensitivity (recall) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return float(sensitivity), float(specificity)


def compute_volume_difference(mask1, mask2, voxel_volume=None):
    """
    Compute volume difference between two masks.

    Args:
        mask1: First mask
        mask2: Second mask
        voxel_volume: Volume of a single voxel (optional, for absolute volume)

    Returns:
        Volume difference (in voxels or mm³ if voxel_volume provided)
    """
    if mask1 is None or mask2 is None:
        return None

    vol1 = int(mask1.sum())
    vol2 = int(mask2.sum())

    diff_voxels = abs(vol1 - vol2)

    if voxel_volume is not None:
        return float(diff_voxels * voxel_volume)

    return float(diff_voxels)


def main():
    """Main function to run the benchmark comparison."""
    # Check if reference mask exists
    if not REFERENCE_MASK.exists():
        print(f"Error: Reference mask not found: {REFERENCE_MASK}", file=sys.stderr)
        print(f"Please ensure the reference segmentation exists.", file=sys.stderr)
        sys.exit(1)

    # Check if Docker mask exists
    if not DOCKER_MASK.exists():
        print(f"Error: Docker output not found: {DOCKER_MASK}", file=sys.stderr)
        print(
            f"Run scripts/run_docker.sh first to generate Docker output.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load masks
    print("Loading masks...")
    reference_mask = load_mask(REFERENCE_MASK)
    docker_mask = load_mask(DOCKER_MASK)

    if reference_mask is None:
        print("Error: Failed to load reference mask", file=sys.stderr)
        sys.exit(1)

    if docker_mask is None:
        print("Error: Failed to load Docker mask", file=sys.stderr)
        sys.exit(1)

    # Ensure masks have the same shape
    if reference_mask.shape != docker_mask.shape:
        print(
            f"Error: Mask shapes don't match: reference {reference_mask.shape} vs "
            f"Docker {docker_mask.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compute metrics
    print("Computing metrics...")
    dice = compute_dice_coefficient(reference_mask, docker_mask)
    hausdorff_95 = compute_hausdorff_distance(
        reference_mask, docker_mask, percentile=95
    )
    sensitivity, specificity = compute_sensitivity_specificity(
        reference_mask, docker_mask
    )

    # Load NIfTI headers to get voxel spacing for volume calculation
    ref_nii = nib.load(str(REFERENCE_MASK))
    voxel_spacing = ref_nii.header.get_zooms()[:3]  # Get x, y, z spacing
    voxel_volume = np.prod(voxel_spacing)  # Volume in mm³

    volume_diff_voxels = compute_volume_difference(reference_mask, docker_mask)
    volume_diff_mm3 = compute_volume_difference(
        reference_mask, docker_mask, voxel_volume
    )

    # Compute volumes
    ref_volume_voxels = reference_mask.sum()
    docker_volume_voxels = docker_mask.sum()
    ref_volume_mm3 = ref_volume_voxels * voxel_volume
    docker_volume_mm3 = docker_volume_voxels * voxel_volume

    # Print benchmark report
    print("\n" + "=" * 70)
    print("SEGMENTATION BENCHMARK REPORT")
    print("=" * 70)
    print(f"Reference mask: {REFERENCE_MASK}")
    print(f"Docker mask:    {DOCKER_MASK}")
    print(f"Mask shape:     {reference_mask.shape}")
    print(f"Voxel spacing:  {voxel_spacing} mm")
    print(f"Voxel volume:   {voxel_volume:.3f} mm³")
    print("-" * 70)
    print("METRICS:")
    print(f"  Dice Coefficient:        {dice:.4f} (range: 0.0-1.0, higher is better)")
    print(f"  Hausdorff Distance (95%): {hausdorff_95:.4f} mm (lower is better)")
    print(
        f"  Sensitivity (Recall):     {sensitivity:.4f} (range: 0.0-1.0, higher is better)"
    )
    print(
        f"  Specificity:              {specificity:.4f} (range: 0.0-1.0, higher is better)"
    )
    print("-" * 70)
    print("VOLUME COMPARISON:")
    print(
        f"  Reference volume:  {ref_volume_voxels:.0f} voxels ({ref_volume_mm3:.2f} mm³)"
    )
    print(
        f"  Docker volume:     {docker_volume_voxels:.0f} voxels ({docker_volume_mm3:.2f} mm³)"
    )
    print(
        f"  Volume difference: {volume_diff_voxels:.0f} voxels ({volume_diff_mm3:.2f} mm³)"
    )
    print(
        f"  Relative difference: {100 * volume_diff_voxels / max(ref_volume_voxels, 1):.2f}%"
    )
    print("=" * 70 + "\n")

    # Validate metrics
    if dice is None:
        print("Error: Dice coefficient computation failed", file=sys.stderr)
        sys.exit(1)

    if hausdorff_95 is None:
        print("Error: Hausdorff distance computation failed", file=sys.stderr)
        sys.exit(1)

    if sensitivity is None or specificity is None:
        print("Error: Sensitivity/Specificity computation failed", file=sys.stderr)
        sys.exit(1)

    # Optional: Check thresholds and warn
    MIN_DICE = 0.7  # At least 70% overlap
    MAX_HAUSDORFF = 15.0  # Maximum 15mm surface distance
    MIN_SENSITIVITY = 0.6  # At least 60% of lesions detected
    MIN_SPECIFICITY = 0.9  # At least 90% specificity

    warnings = []
    if dice < MIN_DICE:
        warnings.append(
            f"Warning: Dice coefficient {dice:.4f} is below threshold {MIN_DICE}. "
            f"Masks have insufficient overlap."
        )

    if hausdorff_95 > MAX_HAUSDORFF:
        warnings.append(
            f"Warning: Hausdorff distance {hausdorff_95:.4f} mm exceeds threshold {MAX_HAUSDORFF} mm. "
            f"Surface distance is too large."
        )

    if sensitivity < MIN_SENSITIVITY:
        warnings.append(
            f"Warning: Sensitivity {sensitivity:.4f} is below threshold {MIN_SENSITIVITY}. "
            f"Too many lesions are missed."
        )

    if specificity < MIN_SPECIFICITY:
        warnings.append(
            f"Warning: Specificity {specificity:.4f} is below threshold {MIN_SPECIFICITY}. "
            f"Too many false positives."
        )

    if warnings:
        print("\n".join(warnings), file=sys.stderr)
        sys.exit(1)

    print("All benchmark metrics passed thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
