"""
Unit tests for src/utils.py functions.
"""
import os
import pytest
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    convert_to_nii,
    get_img_shape,
    save_nii,
    extract_brain,
    check_gpu_memory,
    register_mri,
    propagate_image,
    get_flair_atlas,
    registration_qc,
    print_completed,
    print_ensemble_message,
    print_run,
    grant_permissions
)


class TestConvertToNii:
    """Tests for convert_to_nii function."""
    
    def test_convert_nifti_file(self, tmp_dir, sample_dwi_path):
        """Test conversion of existing NIfTI file."""
        output_path, nii_flag = convert_to_nii(sample_dwi_path, tmp_dir, 'dwi')
        assert nii_flag is True
        assert os.path.exists(output_path)
        assert output_path.endswith('.nii.gz')
    
    def test_convert_nifti_gz_file(self, tmp_dir, sample_dwi_path):
        """Test conversion of .nii.gz file."""
        output_path, nii_flag = convert_to_nii(sample_dwi_path, tmp_dir, 'adc')
        assert nii_flag is True
        assert os.path.exists(output_path)
    
    def test_convert_dicom_directory(self, tmp_dir, mock_dcm2niix):
        """Test conversion of DICOM directory."""
        dicom_dir = os.path.join(tmp_dir, 'dicom')
        os.makedirs(dicom_dir, exist_ok=True)
        # Create a fake .dcm file
        with open(os.path.join(dicom_dir, 'test.dcm'), 'w') as f:
            f.write('fake dicom')
        
        # Mock glob to find .dcm files
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = [os.path.join(dicom_dir, 'test.dcm')]
            # Mock the output file creation
            with patch('os.rename'):
                output_path, nii_flag = convert_to_nii(dicom_dir, tmp_dir, 'dwi')
                assert nii_flag is False
                mock_dcm2niix.assert_called_once()
    
    def test_convert_invalid_file(self, tmp_dir):
        """Test error handling for invalid file format."""
        invalid_path = os.path.join(tmp_dir, 'test.txt')
        with open(invalid_path, 'w') as f:
            f.write('invalid')
        
        with pytest.raises(ValueError, match="No .nii, .nii.gz, .mha, or Dicom files"):
            convert_to_nii(invalid_path, tmp_dir, 'dwi')


class TestGetImgShape:
    """Tests for get_img_shape function."""
    
    def test_get_3d_shape(self, sample_dwi_path):
        """Test getting shape of 3D image."""
        shape = get_img_shape(sample_dwi_path)
        assert shape == 3
    
    def test_get_4d_shape(self, sample_4d_dwi_path):
        """Test getting shape of 4D image."""
        shape = get_img_shape(sample_4d_dwi_path)
        assert shape == 4


class TestSaveNii:
    """Tests for save_nii function."""
    
    def test_save_nii_file(self, tmp_dir, sample_dwi_path):
        """Test saving NIfTI file."""
        nii_img = nib.load(sample_dwi_path)
        data = nii_img.get_fdata()
        affine = nii_img.affine
        header = nii_img.header
        
        output_path = os.path.join(tmp_dir, 'saved.nii.gz')
        save_nii(data, affine, header, output_path)
        
        assert os.path.exists(output_path)
        loaded = nib.load(output_path)
        assert np.allclose(loaded.get_fdata(), data)
        assert np.allclose(loaded.affine, affine)


class TestExtractBrain:
    """Tests for extract_brain function."""
    
    def test_extract_brain_gpu(self, sample_dwi_path, tmp_dir, mock_hd_bet):
        """Test skull stripping with GPU."""
        output_path = os.path.join(tmp_dir, 'brain')
        extract_brain(sample_dwi_path, output_path, gpu=True, save_mask=1)
        mock_hd_bet.assert_called_once()
    
    def test_extract_brain_cpu(self, sample_dwi_path, tmp_dir, mock_hd_bet):
        """Test skull stripping with CPU."""
        output_path = os.path.join(tmp_dir, 'brain')
        extract_brain(sample_dwi_path, output_path, gpu=False, save_mask=0)
        mock_hd_bet.assert_called_once()


class TestCheckGpuMemory:
    """Tests for check_gpu_memory function."""
    
    def test_check_gpu_sufficient_memory(self, mock_gpu_available):
        """Test GPU check with sufficient memory."""
        result = check_gpu_memory(min_free_memory_gb=12)
        assert result is True
    
    def test_check_gpu_insufficient_memory(self, mock_gpu_insufficient_memory):
        """Test GPU check with insufficient memory."""
        result = check_gpu_memory(min_free_memory_gb=12)
        assert result is False
    
    def test_check_gpu_unavailable(self):
        """Test GPU check when GPU is unavailable."""
        with patch('subprocess.run') as mock_run:
            # Mock subprocess.CalledProcessError for nvidia-smi failure
            mock_run.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi', stderr='nvidia-smi not found')
            result = check_gpu_memory(min_free_memory_gb=12)
            assert result is False


class TestRegisterMri:
    """Tests for register_mri function."""
    
    def test_register_mri_rigid(self, sample_dwi_path, sample_flair_path, tmp_dir, mock_sitk_elastix):
        """Test rigid image registration."""
        output_path = os.path.join(tmp_dir, 'registered.nii.gz')
        register_mri(sample_dwi_path, sample_flair_path, output_path, transformation='rigid')
        # Verify ElastixImageFilter was instantiated
        assert mock_sitk_elastix['elastix'].Execute.called
        # Verify ReadImage was called for both images
        assert mock_sitk_elastix['read'].call_count == 2
        # Verify WriteImage was called
        assert mock_sitk_elastix['write'].called


class TestPropagateImage:
    """Tests for propagate_image function."""
    
    def test_propagate_image_mask(self, sample_mask_path, tmp_dir, mock_sitk_read_write):
        """Test propagating a mask image."""
        output_path = os.path.join(tmp_dir, 'propagated_mask.nii.gz')
        
        # Mock transform parameter files
        transform_dir = tmp_dir
        os.makedirs(transform_dir, exist_ok=True)
        param_file = os.path.join(transform_dir, 'TransformParameters.0.txt')
        with open(param_file, 'w') as f:
            f.write('fake transform')
        
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = [param_file]
            with patch('src.utils.sitk.ReadParameterFile', create=True) as mock_read_param, \
                 patch('src.utils.sitk.TransformixImageFilter', create=True) as mock_transformix:
                mock_read_param.return_value = {'ResampleInterpolator': ['FinalNearestNeighborInterpolator']}
                mock_transformix_instance = Mock()
                mock_transformix_instance.GetResultImage.return_value = mock_sitk_read_write['image']
                mock_transformix.return_value = mock_transformix_instance
                
                propagate_image(sample_mask_path, output_path, is_mask=True)
                mock_transformix_instance.Execute.assert_called_once()


class TestGetFlairAtlas:
    """Tests for get_flair_atlas function."""
    
    def test_get_flair_atlas_download(self, tmp_dir, mock_requests_get):
        """Test downloading FLAIR atlas."""
        output_path = os.path.join(tmp_dir, 'atlas.nii.gz')
        get_flair_atlas(output_path)
        mock_requests_get.assert_called_once()
        assert os.path.exists(output_path)
    
    def test_get_flair_atlas_exists(self, tmp_dir, mock_requests_get):
        """Test that atlas is not re-downloaded if it exists."""
        output_path = os.path.join(tmp_dir, 'atlas.nii.gz')
        # Create existing file
        with open(output_path, 'wb') as f:
            f.write(b'existing atlas')
        
        get_flair_atlas(output_path)
        mock_requests_get.assert_not_called()


class TestRegistrationQc:
    """Tests for registration_qc function."""
    
    def test_registration_qc_basic(self, sample_dwi_path, sample_adc_path, sample_mask_path, tmp_dir):
        """Test basic QC image generation."""
        output_path = os.path.join(tmp_dir, 'qc.png')
        image_paths = [sample_dwi_path, sample_adc_path]
        labels = ['dwi', 'adc']
        
        registration_qc(image_paths, labels, output_path, sample_mask_path)
        assert os.path.exists(output_path)
    
    def test_registration_qc_with_brain_mask(self, sample_dwi_path, sample_adc_path, 
                                            sample_mask_path, sample_flair_path, tmp_dir):
        """Test QC image generation with brain mask."""
        output_path = os.path.join(tmp_dir, 'qc.png')
        image_paths = [sample_dwi_path, sample_adc_path, sample_flair_path]
        labels = ['dwi', 'adc', 'flair']
        
        registration_qc(image_paths, labels, output_path, sample_mask_path, 
                       brain_mask_path=sample_mask_path)
        assert os.path.exists(output_path)


class TestPrintFunctions:
    """Tests for print utility functions."""
    
    def test_print_completed(self, capsys):
        """Test print_completed function."""
        print_completed('/path/to/file', '/path/to/output')
        captured = capsys.readouterr()
        assert 'Finished:' in captured.out
        assert '/path/to/file' in captured.out
        assert '/path/to/output' in captured.out
    
    def test_print_ensemble_message(self, capsys):
        """Test print_ensemble_message function."""
        print_ensemble_message()
        captured = capsys.readouterr()
        assert 'DeepISLES' in captured.out or 'Ensemble' in captured.out or 'cite' in captured.out
    
    def test_print_run(self, capsys):
        """Test print_run function."""
        print_run('SEALS')
        captured = capsys.readouterr()
        assert 'SEALS' in captured.out


class TestGrantPermissions:
    """Tests for grant_permissions function."""
    
    def test_grant_permissions(self, tmp_dir):
        """Test granting permissions to directory."""
        test_dir = os.path.join(tmp_dir, 'test_dir')
        os.makedirs(test_dir, exist_ok=True)
        
        grant_permissions(test_dir)
        # Should not raise an error
        assert os.path.exists(test_dir)

