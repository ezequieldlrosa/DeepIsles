"""
Helper functions to generate synthetic NIfTI test images for testing.
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path


def create_test_nifti_image(
    shape=(64, 64, 64),
    dtype=np.float32,
    affine=None,
    output_path=None,
    data=None
):
    """
    Create a synthetic NIfTI image for testing.
    
    Args:
        shape: Tuple of image dimensions (default: (64, 64, 64))
        dtype: NumPy data type (default: np.float32)
        affine: Affine transformation matrix (default: identity)
        output_path: Optional path to save the image
        data: Optional numpy array (if None, random data is generated)
    
    Returns:
        nibabel.Nifti1Image: The created NIfTI image
    """
    if affine is None:
        affine = np.eye(4)
    
    if data is None:
        data = np.random.rand(*shape).astype(dtype) * 100
    
    nii_img = nib.Nifti1Image(data, affine)
    
    if output_path:
        nib.save(nii_img, output_path)
    
    return nii_img


def create_test_dwi_image(output_path=None, shape=(64, 64, 64)):
    """Create a test DWI image."""
    data = np.random.rand(*shape).astype(np.float32) * 1000
    return create_test_nifti_image(shape=shape, data=data, output_path=output_path)


def create_test_adc_image(output_path=None, shape=(64, 64, 64)):
    """Create a test ADC image."""
    data = np.random.rand(*shape).astype(np.float32) * 2000
    return create_test_nifti_image(shape=shape, data=data, output_path=output_path)


def create_test_flair_image(output_path=None, shape=(64, 64, 64)):
    """Create a test FLAIR image."""
    data = np.random.rand(*shape).astype(np.float32) * 500
    return create_test_nifti_image(shape=shape, data=data, output_path=output_path)


def create_test_mask(output_path=None, shape=(64, 64, 64), lesion_region=None):
    """
    Create a test binary mask.
    
    Args:
        output_path: Optional path to save the mask
        shape: Tuple of image dimensions
        lesion_region: Optional tuple (x, y, z, radius) defining lesion location
    
    Returns:
        nibabel.Nifti1Image: The created mask image
    """
    mask_data = np.zeros(shape, dtype=np.uint8)
    
    if lesion_region:
        x, y, z, radius = lesion_region
        # Create a simple spherical lesion
        coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((coords[0] - x)**2 + (coords[1] - y)**2 + (coords[2] - z)**2)
        mask_data[dist < radius] = 1
    else:
        # Create a simple lesion in the center
        center = tuple(s // 2 for s in shape)
        radius = min(shape) // 4
        coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt(
            (coords[0] - center[0])**2 + 
            (coords[1] - center[1])**2 + 
            (coords[2] - center[2])**2
        )
        mask_data[dist < radius] = 1
    
    return create_test_nifti_image(shape=shape, data=mask_data, output_path=output_path)


def create_test_4d_dwi_image(output_path=None, shape=(64, 64, 64, 2)):
    """Create a test 4D DWI image with 2 volumes."""
    data = np.random.rand(*shape).astype(np.float32) * 1000
    return create_test_nifti_image(shape=shape, data=data, output_path=output_path)


def create_test_prediction_array(shape=(64, 64, 64), algorithm='seals'):
    """
    Create a test prediction array for an algorithm.
    
    Args:
        shape: Tuple of image dimensions
        algorithm: Algorithm name ('seals', 'nvauto', 'factorizer')
    
    Returns:
        numpy.ndarray: Prediction array (binary mask)
    """
    pred = np.zeros(shape, dtype=np.int8)
    # Create a simple lesion prediction
    center = tuple(s // 2 for s in shape)
    radius = min(shape) // 4
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (coords[0] - center[0])**2 + 
        (coords[1] - center[1])**2 + 
        (coords[2] - center[2])**2
    )
    pred[dist < radius] = 1
    
    # Add some noise/variation based on algorithm
    if algorithm == 'nvauto':
        # Slightly different shape
        pred[dist < radius * 0.9] = 1
    elif algorithm == 'factorizer':
        # Slightly larger
        pred[dist < radius * 1.1] = 1
    
    return pred

