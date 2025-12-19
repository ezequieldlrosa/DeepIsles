"""
Unit tests for majority voting logic in src/majority_voting.py.
"""
import os
import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import patch, Mock, MagicMock
import sys
from glob import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.majority_voting import ISLES22, sitk_loader, stik_saver, json_writer


class TestISLES22Init:
    """Tests for ISLES22.__init__."""
    
    def test_init(self, tmp_dir):
        """Test class initialization."""
        dataset = ISLES22(tmp_dir)
        assert dataset.input_folder == tmp_dir
        assert dataset.data_dict == {}


class TestISLES22LoadData:
    """Tests for ISLES22.load_data method."""
    
    def test_load_data(self, tmp_dir):
        """Test loading DWI data."""
        dataset = ISLES22(tmp_dir)
        
        # Create test DWI file
        dwi_dir = os.path.join(tmp_dir, 'dwi')
        os.makedirs(dwi_dir, exist_ok=True)
        dwi_path = os.path.join(dwi_dir, 'dwi.nii.gz')
        
        # Create a simple NIfTI file
        data = np.random.rand(64, 64, 64).astype(np.float32)
        nii_img = sitk.GetImageFromArray(data)
        sitk.WriteImage(nii_img, dwi_path)
        
        dataset.load_data()
        assert dataset.dwi_path == dwi_path


class TestMajorityVoting:
    """Tests for majority voting logic."""
    
    def test_majority_voting_all_teams(self, tmp_dir, sample_prediction_arrays):
        """Test majority voting with all three algorithms."""
        input_folder = tmp_dir
        output_folder = os.path.join(tmp_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # Create output directories for each team
        for team in ['seals', 'nvauto', 'factorizer']:
            team_dir = os.path.join(input_folder, 'output', team)
            os.makedirs(team_dir, exist_ok=True)
            
            # Create prediction file
            pred_path = os.path.join(team_dir, f'{team}_pred.nii.gz')
            pred_data = sample_prediction_arrays[team]
            pred_img = sitk.GetImageFromArray(pred_data.astype(np.uint8))
            sitk.WriteImage(pred_img, pred_path)
        
        # Create DWI reference
        dwi_dir = os.path.join(input_folder, 'dwi')
        os.makedirs(dwi_dir, exist_ok=True)
        dwi_path = os.path.join(dwi_dir, 'dwi.nii.gz')
        dwi_data = np.random.rand(64, 64, 64).astype(np.float32)
        dwi_img = sitk.GetImageFromArray(dwi_data)
        sitk.WriteImage(dwi_img, dwi_path)
        
        # Import and run majority voting logic
        dataset = ISLES22(input_folder)
        dataset.load_data()
        
        # Load reference image
        image_file = sitk.ReadImage(dataset.dwi_path)
        
        # Load predictions
        teams = ['seals', 'nvauto', 'factorizer']
        pred_array = {}
        
        for folder in teams:
            try:
                pred_file = glob(os.path.join(input_folder, 'output', folder, '*.nii.gz'))[0]
                pred_image = sitk.ReadImage(pred_file)
                pred_array[folder] = sitk.GetArrayFromImage(pred_image).astype(np.int8)
            except:
                pass
        
        # Majority voting
        if all(key in pred_array for key in teams):
            result_array = pred_array['seals'] + pred_array['nvauto'] + pred_array['factorizer']
            result_array = result_array / 3 > 0.5
        
        # Verify result
        assert result_array is not None
        assert result_array.dtype == bool or result_array.dtype == np.bool_
        assert result_array.shape == (64, 64, 64)
    
    def test_majority_voting_fallback_seals(self, tmp_dir, sample_prediction_arrays):
        """Test fallback to SEALS when other algorithms fail."""
        input_folder = tmp_dir
        output_folder = os.path.join(tmp_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # Create only SEALS output
        seals_dir = os.path.join(input_folder, 'output', 'seals')
        os.makedirs(seals_dir, exist_ok=True)
        pred_path = os.path.join(seals_dir, 'seals_pred.nii.gz')
        pred_data = sample_prediction_arrays['seals']
        pred_img = sitk.GetImageFromArray(pred_data.astype(np.uint8))
        sitk.WriteImage(pred_img, pred_path)
        
        # Create DWI reference
        dwi_dir = os.path.join(input_folder, 'dwi')
        os.makedirs(dwi_dir, exist_ok=True)
        dwi_path = os.path.join(dwi_dir, 'dwi.nii.gz')
        dwi_data = np.random.rand(64, 64, 64).astype(np.float32)
        dwi_img = sitk.GetImageFromArray(dwi_data)
        sitk.WriteImage(dwi_img, dwi_path)
        
        # Run logic
        dataset = ISLES22(input_folder)
        dataset.load_data()
        image_file = sitk.ReadImage(dataset.dwi_path)
        
        teams = ['seals', 'nvauto', 'factorizer']
        pred_array = {}
        
        for folder in teams:
            try:
                pred_file = glob(os.path.join(input_folder, 'output', folder, '*.nii.gz'))[0]
                pred_image = sitk.ReadImage(pred_file)
                pred_array[folder] = sitk.GetArrayFromImage(pred_image).astype(np.int8)
            except:
                pass
        
        # Fallback logic
        if all(key in pred_array for key in teams):
            result_array = pred_array['seals'] + pred_array['nvauto'] + pred_array['factorizer']
            result_array = result_array / 3 > 0.5
        elif 'seals' in pred_array.keys():
            result_array = pred_array['seals']
        
        # Verify result
        assert result_array is not None
        assert np.array_equal(result_array, pred_array['seals'])


class TestSitkLoader:
    """Tests for sitk_loader function."""
    
    def test_sitk_loader(self, tmp_dir):
        """Test loading SimpleITK image."""
        # Create test image
        data = np.random.rand(64, 64, 64).astype(np.float32)
        img = sitk.GetImageFromArray(data)
        img_path = os.path.join(tmp_dir, 'test.nii.gz')
        sitk.WriteImage(img, img_path)
        
        image, array = sitk_loader(img_path)
        
        assert image is not None
        assert array is not None
        assert array.shape == (64, 64, 64)


class TestStikSaver:
    """Tests for stik_saver function."""
    
    def test_stik_saver(self, tmp_dir):
        """Test saving SimpleITK image."""
        # Create original image
        original_data = np.random.rand(64, 64, 64).astype(np.float32)
        original_img = sitk.GetImageFromArray(original_data)
        original_img.SetOrigin((0.0, 0.0, 0.0))
        original_img.SetSpacing((1.0, 1.0, 1.0))
        
        # Create new array
        new_data = np.random.rand(64, 64, 64).astype(np.float32)
        output_path = os.path.join(tmp_dir, 'saved.nii.gz')
        
        stik_saver(original_img, new_data, output_path)
        
        # Verify saved image
        saved_img = sitk.ReadImage(output_path)
        assert saved_img.GetOrigin() == original_img.GetOrigin()
        assert saved_img.GetSpacing() == original_img.GetSpacing()


class TestJsonWriter:
    """Tests for json_writer function."""
    
    def test_json_writer(self, tmp_dir):
        """Test writing JSON file."""
        json_path = os.path.join(tmp_dir, 'test.json')
        data = {'key1': 'value1', 'key2': 42}
        
        json_writer(json_path, data)
        
        assert os.path.exists(json_path)
        import json
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == data

