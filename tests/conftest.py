"""
Shared pytest fixtures and configuration.
"""
import os
import tempfile
import shutil
import gc
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from tests.fixtures.sample_images import (
    create_test_dwi_image,
    create_test_adc_image,
    create_test_flair_image,
    create_test_mask,
    create_test_4d_dwi_image,
    create_test_prediction_array
)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Force garbage collection after each test to prevent memory accumulation."""
    yield
    gc.collect()
    plt.close('all')  # Close any lingering matplotlib figures


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_dwi_path(tmp_dir):
    """Create a sample DWI NIfTI file."""
    path = os.path.join(tmp_dir, 'dwi.nii.gz')
    create_test_dwi_image(output_path=path)
    return path


@pytest.fixture
def sample_adc_path(tmp_dir):
    """Create a sample ADC NIfTI file."""
    path = os.path.join(tmp_dir, 'adc.nii.gz')
    create_test_adc_image(output_path=path)
    return path


@pytest.fixture
def sample_flair_path(tmp_dir):
    """Create a sample FLAIR NIfTI file."""
    path = os.path.join(tmp_dir, 'flair.nii.gz')
    create_test_flair_image(output_path=path)
    return path


@pytest.fixture
def sample_mask_path(tmp_dir):
    """Create a sample binary mask file."""
    path = os.path.join(tmp_dir, 'mask.nii.gz')
    create_test_mask(output_path=path)
    return path


@pytest.fixture
def sample_4d_dwi_path(tmp_dir):
    """Create a sample 4D DWI NIfTI file."""
    path = os.path.join(tmp_dir, 'dwi_4d.nii.gz')
    create_test_4d_dwi_image(output_path=path)
    return path


@pytest.fixture
def ensemble_path(tmp_dir):
    """Create a mock ensemble path structure."""
    ensemble_dir = os.path.join(tmp_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, 'src', 'SEALS'), exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, 'src', 'NVAUTO'), exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, 'src', 'FACTORIZER'), exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, 'src', 'HD-BET'), exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, 'weights'), exist_ok=True)
    return ensemble_dir


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run calls."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')
        yield mock_run


@pytest.fixture
def mock_subprocess_call():
    """Mock subprocess.call calls."""
    with patch('subprocess.call') as mock_call:
        mock_call.return_value = 0
        yield mock_call


@pytest.fixture
def mock_subprocess_popen():
    """Mock subprocess.Popen calls."""
    with patch('subprocess.Popen') as mock_popen:
        mock_process = Mock()
        mock_process.communicate.return_value = ('', '')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        yield mock_popen


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability check."""
    with patch('subprocess.run') as mock_run:
        # Mock nvidia-smi output with sufficient memory (16GB)
        mock_run.return_value = Mock(
            returncode=0,
            stdout='16384\n',  # 16GB in MB
            stderr=''
        )
        yield mock_run


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability."""
    with patch('subprocess.run') as mock_run:
        # Mock nvidia-smi failure or insufficient memory
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        yield mock_run


@pytest.fixture
def mock_gpu_insufficient_memory():
    """Mock GPU with insufficient memory."""
    with patch('subprocess.run') as mock_run:
        # Mock nvidia-smi output with insufficient memory (8GB)
        mock_run.return_value = Mock(
            returncode=0,
            stdout='8192\n',  # 8GB in MB
            stderr=''
        )
        yield mock_run


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for downloading atlas."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.content = b'fake atlas data'
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_sitk_elastix():
    """Mock SimpleITK Elastix operations."""
    # Patch in the namespace where SimpleITK is used (src.utils)
    # Note: ElastixImageFilter may come from SimpleITK-SimpleElastix, so we patch it
    # where it's accessed in the utils module with create=True
    with patch('src.utils.sitk.ElastixImageFilter', create=True) as mock_elastix_class, \
         patch('src.utils.sitk.ReadImage') as mock_read, \
         patch('src.utils.sitk.WriteImage') as mock_write, \
         patch('src.utils.sitk.GetDefaultParameterMap', create=True) as mock_get_param:
        
        mock_elastix = Mock()
        mock_elastix_class.return_value = mock_elastix
        
        # Mock result image
        mock_result = Mock()
        mock_elastix.GetResultImage.return_value = mock_result
        mock_elastix.Execute.return_value = None
        
        # Mock ReadImage to return a mock image
        mock_image = Mock()
        mock_read.return_value = mock_image
        
        # Mock GetDefaultParameterMap to return a parameter map
        mock_get_param.return_value = {'Transformation': ['RigidTransform']}
        
        yield {
            'elastix': mock_elastix,
            'read': mock_read,
            'write': mock_write,
            'get_param': mock_get_param
        }


@pytest.fixture
def mock_sitk_read_write():
    """Mock SimpleITK ReadImage and WriteImage."""
    with patch('SimpleITK.ReadImage') as mock_read, \
         patch('SimpleITK.WriteImage') as mock_write:
        
        # Create a mock image
        mock_image = Mock()
        mock_image.GetOrigin.return_value = (0.0, 0.0, 0.0)
        mock_image.GetSpacing.return_value = (1.0, 1.0, 1.0)
        mock_image.GetDirection.return_value = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        
        mock_read.return_value = mock_image
        mock_write.return_value = None
        
        yield {'read': mock_read, 'write': mock_write, 'image': mock_image}


@pytest.fixture
def mock_hd_bet():
    """Mock hd-bet command."""
    with patch('subprocess.call') as mock_call:
        mock_call.return_value = 0
        yield mock_call


@pytest.fixture
def mock_dcm2niix():
    """Mock dcm2niix command."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)
        yield mock_run


@pytest.fixture
def sample_prediction_arrays():
    """Create sample prediction arrays for all three algorithms."""
    return {
        'seals': create_test_prediction_array(algorithm='seals'),
        'nvauto': create_test_prediction_array(algorithm='nvauto'),
        'factorizer': create_test_prediction_array(algorithm='factorizer')
    }


@pytest.fixture
def mock_nnunet_launcher():
    """Mock nnunet launcher script."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)
        yield mock_run


@pytest.fixture
def output_dir(tmp_dir):
    """Create an output directory for tests."""
    output = os.path.join(tmp_dir, 'output')
    os.makedirs(output, exist_ok=True)
    return output


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

