"""
Unit tests for IslesEnsemble class in src/isles22_ensemble.py.
"""
import os
import pytest
import numpy as np
import nibabel as nib
from unittest.mock import patch, Mock, MagicMock
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.isles22_ensemble import IslesEnsemble


class TestIslesEnsembleInit:
    """Tests for IslesEnsemble.__init__."""
    
    def test_init(self):
        """Test class initialization."""
        ensemble = IslesEnsemble()
        assert ensemble is not None


class TestIslesEnsembleCheckImages:
    """Tests for IslesEnsemble.check_images method."""
    
    def test_check_images_3d_valid(self, sample_dwi_path, sample_adc_path):
        """Test checking valid 3D images."""
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = sample_dwi_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = None
        
        # Should not raise
        ensemble.check_images()
    
    def test_check_images_4d_dwi_single_volume(self, tmp_dir, sample_adc_path):
        """Test handling 4D DWI with single volume."""
        # Create 4D DWI with shape (64, 64, 64, 1)
        dwi_4d_path = os.path.join(tmp_dir, 'dwi_4d.nii.gz')
        data = np.random.rand(64, 64, 64, 1).astype(np.float32) * 1000
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, dwi_4d_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = dwi_4d_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = None
        
        ensemble.check_images()
        # Should convert to 3D
        assert len(nib.load(ensemble.input_dwi_path).shape) == 3
    
    def test_check_images_4d_dwi_two_volumes(self, tmp_dir, sample_adc_path):
        """Test handling 4D DWI with two volumes."""
        # Create 4D DWI with shape (64, 64, 64, 2)
        dwi_4d_path = os.path.join(tmp_dir, 'dwi_4d.nii.gz')
        data = np.random.rand(64, 64, 64, 2).astype(np.float32) * 1000
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, dwi_4d_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = dwi_4d_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = None
        
        ensemble.check_images()
        # Should extract last volume and convert to 3D
        assert len(nib.load(ensemble.input_dwi_path).shape) == 3
    
    def test_check_images_4d_dwi_invalid(self, tmp_dir, sample_adc_path):
        """Test error for 4D DWI with too many volumes."""
        # Create 4D DWI with shape (64, 64, 64, 5)
        dwi_4d_path = os.path.join(tmp_dir, 'dwi_4d.nii.gz')
        data = np.random.rand(64, 64, 64, 5).astype(np.float32) * 1000
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, dwi_4d_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = dwi_4d_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = None
        
        with pytest.raises(ValueError, match="DWI is 4D and contains 5 volumes"):
            ensemble.check_images()
    
    def test_check_images_invalid_dimension(self, tmp_dir, sample_adc_path):
        """Test error for invalid image dimension."""
        # Create 2D image
        dwi_2d_path = os.path.join(tmp_dir, 'dwi_2d.nii.gz')
        data = np.random.rand(64, 64).astype(np.float32) * 1000
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, dwi_2d_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = dwi_2d_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = None
        
        with pytest.raises(ValueError, match="DWI is 2D"):
            ensemble.check_images()
    
    def test_check_images_affine_mismatch(self, tmp_dir, sample_dwi_path):
        """Test handling of affine matrix mismatch."""
        # Create ADC with different affine
        adc_path = os.path.join(tmp_dir, 'adc.nii.gz')
        data = np.random.rand(64, 64, 64).astype(np.float32) * 2000
        different_affine = np.eye(4)
        different_affine[0, 0] = 2.0  # Different spacing
        nii_img = nib.Nifti1Image(data, different_affine)
        nib.save(nii_img, adc_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = sample_dwi_path
        ensemble.input_adc_path = adc_path
        ensemble.input_flair_path = None
        
        with pytest.warns(UserWarning, match="DWI and ADC have different affine matrices"):
            ensemble.check_images()
        
        # Affine should be updated to match DWI
        adc_loaded = nib.load(ensemble.input_adc_path)
        dwi_loaded = nib.load(sample_dwi_path)
        assert np.array_equal(adc_loaded.affine, dwi_loaded.affine)
    
    def test_check_images_adc_not_3d(self, sample_dwi_path, tmp_dir):
        """Test error when ADC is not 3D."""
        # Create 2D ADC
        adc_2d_path = os.path.join(tmp_dir, 'adc_2d.nii.gz')
        data = np.random.rand(64, 64).astype(np.float32) * 2000
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, adc_2d_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = sample_dwi_path
        ensemble.input_adc_path = adc_2d_path
        ensemble.input_flair_path = None
        
        with pytest.raises(AssertionError, match="ADC is not 3D"):
            ensemble.check_images()
    
    def test_check_images_flair_not_3d(self, sample_dwi_path, sample_adc_path, tmp_dir):
        """Test error when FLAIR is not 3D."""
        # Create 2D FLAIR
        flair_2d_path = os.path.join(tmp_dir, 'flair_2d.nii.gz')
        data = np.random.rand(64, 64).astype(np.float32) * 500
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, flair_2d_path)
        
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = sample_dwi_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = flair_2d_path
        
        with pytest.raises(AssertionError, match="FLAIR is not 3D"):
            ensemble.check_images()


class TestIslesEnsembleLoadImages:
    """Tests for IslesEnsemble.load_images method."""
    
    def test_load_images_nifti(self, sample_dwi_path, sample_adc_path, tmp_dir):
        """Test loading NIfTI images."""
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = sample_dwi_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = None
        ensemble.tmp_out_dir = tmp_dir
        ensemble.skull_strip = False  # Initialize skull_strip attribute
        
        with patch('src.isles22_ensemble.convert_to_nii') as mock_convert:
            mock_convert.side_effect = [
                (sample_dwi_path, True),  # DWI already NIfTI
                (sample_adc_path, True),  # ADC already NIfTI
            ]
            ensemble.load_images()
            
            assert mock_convert.call_count == 2
            assert ensemble.input_dwi_path == sample_dwi_path
            assert ensemble.input_adc_path == sample_adc_path
    
    def test_load_images_with_flair(self, sample_dwi_path, sample_adc_path, 
                                    sample_flair_path, tmp_dir):
        """Test loading images including FLAIR."""
        ensemble = IslesEnsemble()
        ensemble.input_dwi_path = sample_dwi_path
        ensemble.input_adc_path = sample_adc_path
        ensemble.input_flair_path = sample_flair_path
        ensemble.tmp_out_dir = tmp_dir
        ensemble.skull_strip = False  # Initialize skull_strip attribute
        
        with patch('src.isles22_ensemble.convert_to_nii') as mock_convert:
            mock_convert.side_effect = [
                (sample_dwi_path, True),
                (sample_adc_path, True),
                (sample_flair_path, True),
            ]
            ensemble.load_images()
            
            assert mock_convert.call_count == 3


class TestIslesEnsembleRunCommand:
    """Tests for IslesEnsemble.run_command static method."""
    
    def test_run_command(self, tmp_dir):
        """Test running a command."""
        command = 'echo "test"'
        IslesEnsemble.run_command(command, tmp_dir)
        # Command should execute without error


class TestIslesEnsembleInference:
    """Tests for IslesEnsemble.inference method."""
    
    def test_inference_fast_mode(self, ensemble_path, tmp_dir, mock_subprocess_run):
        """Test inference in fast mode (SEALS only)."""
        ensemble = IslesEnsemble()
        ensemble.ensemble_path = ensemble_path
        ensemble.tmp_out_dir = tmp_dir
        ensemble.weights_dir = os.path.join(ensemble_path, 'weights')
        ensemble.fast = True
        ensemble.input_flair_path = None
        ensemble.parallelize = False
        
        # Create mock launcher script
        launcher_path = os.path.join(ensemble_path, 'src', 'SEALS', 'nnunet_launcher.sh')
        os.makedirs(os.path.dirname(launcher_path), exist_ok=True)
        with open(launcher_path, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        os.chmod(launcher_path, 0o755)
        
        ensemble.inference()
        # Should only call SEALS
        assert mock_subprocess_run.called
    
    def test_inference_full_mode(self, ensemble_path, tmp_dir, sample_flair_path, 
                                mock_subprocess_run):
        """Test inference in full mode (all algorithms)."""
        ensemble = IslesEnsemble()
        ensemble.ensemble_path = ensemble_path
        ensemble.tmp_out_dir = tmp_dir
        ensemble.weights_dir = os.path.join(ensemble_path, 'weights')
        ensemble.fast = False
        ensemble.input_flair_path = sample_flair_path
        ensemble.parallelize = False
        
        # Create mock scripts
        for subdir in ['SEALS', 'NVAUTO', 'FACTORIZER']:
            script_dir = os.path.join(ensemble_path, 'src', subdir)
            os.makedirs(script_dir, exist_ok=True)
            if subdir == 'SEALS':
                launcher_path = os.path.join(script_dir, 'nnunet_launcher.sh')
                with open(launcher_path, 'w') as f:
                    f.write('#!/bin/bash\necho "test"')
                os.chmod(launcher_path, 0o755)
            else:
                script_path = os.path.join(script_dir, 'process.py')
                with open(script_path, 'w') as f:
                    f.write('print("test")')
        
        ensemble.inference()
        # Should call all three algorithms
        assert mock_subprocess_run.called


class TestIslesEnsembleCopyOutputClean:
    """Tests for IslesEnsemble.copy_output_clean method."""
    
    def test_copy_output_clean_no_options(self, tmp_dir):
        """Test cleaning output without saving team outputs or MNI."""
        ensemble = IslesEnsemble()
        # Use separate directories for tmp and output to avoid deletion issues
        ensemble.tmp_out_dir = os.path.join(tmp_dir, 'tmp_work')
        ensemble.output_path = os.path.join(tmp_dir, 'final_output')
        ensemble.ensemble_output_path = os.path.join(ensemble.tmp_out_dir, 'output', 'deepisles')
        ensemble.ensemble_path = tmp_dir
        ensemble.save_team_outputs = False
        ensemble.results_mni = False
        ensemble.keep_tmp_files = False
        
        # Create ensemble output directory with results
        os.makedirs(ensemble.ensemble_output_path, exist_ok=True)
        with open(os.path.join(ensemble.ensemble_output_path, 'lesion_msk.nii.gz'), 'w') as f:
            f.write('test')
        with open(os.path.join(ensemble.ensemble_output_path, 'output_screenshot.png'), 'w') as f:
            f.write('test')
        
        ensemble.copy_output_clean()
        # Output files should exist
        assert os.path.exists(os.path.join(ensemble.output_path, 'lesion_msk.nii.gz'))
        assert os.path.exists(os.path.join(ensemble.output_path, 'output_screenshot.png'))
    
    def test_copy_output_clean_save_team_outputs(self, tmp_dir):
        """Test saving team outputs."""
        ensemble = IslesEnsemble()
        # Use separate directories for tmp and output to avoid deletion issues
        ensemble.tmp_out_dir = os.path.join(tmp_dir, 'tmp_work')
        ensemble.output_path = os.path.join(tmp_dir, 'final_output')
        ensemble.ensemble_output_path = os.path.join(ensemble.tmp_out_dir, 'output', 'deepisles')
        ensemble.ensemble_path = tmp_dir  # Required for save_team_outputs
        ensemble.save_team_outputs = True
        ensemble.results_mni = False
        ensemble.keep_tmp_files = False
        
        os.makedirs(ensemble.tmp_out_dir, exist_ok=True)
        os.makedirs(ensemble.output_path, exist_ok=True)
        team_output_dir = os.path.join(ensemble.tmp_out_dir, 'output')
        os.makedirs(team_output_dir, exist_ok=True)
        # Create a test subfolder in output (excluding 'deepisles')
        test_folder = os.path.join(team_output_dir, 'test_team')
        os.makedirs(test_folder, exist_ok=True)
        with open(os.path.join(test_folder, 'test.txt'), 'w') as f:
            f.write('test')
        
        # Create ensemble output
        os.makedirs(ensemble.ensemble_output_path, exist_ok=True)
        with open(os.path.join(ensemble.ensemble_output_path, 'lesion_msk.nii.gz'), 'w') as f:
            f.write('test')
        with open(os.path.join(ensemble.ensemble_output_path, 'output_screenshot.png'), 'w') as f:
            f.write('test')
        
        ensemble.copy_output_clean()
        # Team outputs should be copied
        assert os.path.exists(os.path.join(ensemble.output_path, 'output_teams', 'test_team'))
        assert os.path.exists(os.path.join(ensemble.output_path, 'output_teams', 'test_team', 'test.txt'))
    
    def test_copy_output_clean_results_mni(self, tmp_dir):
        """Test saving MNI results."""
        ensemble = IslesEnsemble()
        # Use separate directories for tmp and output to avoid deletion issues
        ensemble.tmp_out_dir = os.path.join(tmp_dir, 'tmp_work')
        ensemble.output_path = os.path.join(tmp_dir, 'final_output')
        ensemble.ensemble_output_path = os.path.join(ensemble.tmp_out_dir, 'output', 'deepisles')
        ensemble.save_team_outputs = False
        ensemble.results_mni = True
        ensemble.keep_tmp_files = False
        
        os.makedirs(ensemble.tmp_out_dir, exist_ok=True)
        os.makedirs(ensemble.output_path, exist_ok=True)
        mni_dir = os.path.join(ensemble.tmp_out_dir, 'mni')
        os.makedirs(mni_dir, exist_ok=True)
        
        # Create ensemble output
        os.makedirs(ensemble.ensemble_output_path, exist_ok=True)
        with open(os.path.join(ensemble.ensemble_output_path, 'lesion_msk.nii.gz'), 'w') as f:
            f.write('test')
        with open(os.path.join(ensemble.ensemble_output_path, 'output_screenshot.png'), 'w') as f:
            f.write('test')
        
        # Create test files
        test_nii = os.path.join(mni_dir, 'test.nii.gz')
        test_png = os.path.join(mni_dir, 'test.png')
        with open(test_nii, 'w') as f:
            f.write('test')
        with open(test_png, 'w') as f:
            f.write('test')
        
        ensemble.copy_output_clean()
        # MNI files should be copied
        assert os.path.exists(os.path.join(ensemble.output_path, 'output_mni', 'test.nii.gz'))
        assert os.path.exists(os.path.join(ensemble.output_path, 'output_mni', 'test.png'))

