"""
Unit tests for main.py CLI interface.
"""
import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import main


class TestMainFunction:
    """Tests for main.main() function."""
    
    def test_main_with_required_args(self, tmp_dir, sample_dwi_path, sample_adc_path):
        """Test main function with required arguments."""
        # Create a mock for IslesEnsemble
        with patch('main.IslesEnsemble') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            # Create the data directory structure
            data_dir = os.path.join(tmp_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            shutil.copy(sample_dwi_path, os.path.join(data_dir, os.path.basename(sample_dwi_path)))
            shutil.copy(sample_adc_path, os.path.join(data_dir, os.path.basename(sample_adc_path)))
            
            # Mock argparse to return test arguments
            with patch('sys.argv', [
                'main.py',
                '--dwi_file_name', os.path.basename(sample_dwi_path),
                '--adc_file_name', os.path.basename(sample_adc_path),
            ]):
                # Use patch.object for os.path.join in main module specifically
                original_join = os.path.join
                
                def mock_join_for_main(*args):
                    if args and args[0] == '/app':
                        return original_join(tmp_dir, *args[1:])
                    return original_join(*args)
                
                with patch.object(main.os.path, 'join', side_effect=mock_join_for_main):
                    with patch.object(main.os.path, 'exists', return_value=True):
                        with patch.object(main, 'subprocess'):
                            try:
                                main.main()
                            except SystemExit:
                                pass  # argparse may raise SystemExit
                            
                            # Verify ensemble was called
                            mock_ensemble.predict_ensemble.assert_called_once()
    
    def test_main_with_flair(self, tmp_dir, sample_dwi_path, sample_adc_path, sample_flair_path):
        """Test main function with FLAIR image."""
        with patch('main.IslesEnsemble') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            # Create the data directory structure
            data_dir = os.path.join(tmp_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            shutil.copy(sample_dwi_path, os.path.join(data_dir, os.path.basename(sample_dwi_path)))
            shutil.copy(sample_adc_path, os.path.join(data_dir, os.path.basename(sample_adc_path)))
            shutil.copy(sample_flair_path, os.path.join(data_dir, os.path.basename(sample_flair_path)))
            
            with patch('sys.argv', [
                'main.py',
                '--dwi_file_name', os.path.basename(sample_dwi_path),
                '--adc_file_name', os.path.basename(sample_adc_path),
                '--flair_file_name', os.path.basename(sample_flair_path),
            ]):
                original_join = os.path.join
                
                def mock_join_for_main(*args):
                    if args and args[0] == '/app':
                        return original_join(tmp_dir, *args[1:])
                    return original_join(*args)
                
                with patch.object(main.os.path, 'join', side_effect=mock_join_for_main):
                    with patch.object(main.os.path, 'exists', return_value=True):
                        with patch.object(main, 'subprocess'):
                            try:
                                main.main()
                            except SystemExit:
                                pass
                            
                            # Verify ensemble was called with FLAIR
                            mock_ensemble.predict_ensemble.assert_called_once()
                            call_kwargs = mock_ensemble.predict_ensemble.call_args[1]
                            assert 'input_flair_path' in call_kwargs
    
    def test_main_missing_dwi(self):
        """Test error when DWI file name is missing."""
        with patch('sys.argv', ['main.py']):
            with pytest.raises((SystemExit, ValueError)):
                main.main()
    
    def test_main_missing_adc(self):
        """Test error when ADC file name is missing."""
        with patch('sys.argv', ['main.py', '--dwi_file_name', 'test.nii.gz']):
            with pytest.raises((SystemExit, ValueError)):
                main.main()
    
    def test_main_file_not_found(self, tmp_dir):
        """Test error when input file does not exist."""
        with patch('sys.argv', [
            'main.py',
            '--dwi_file_name', 'nonexistent.nii.gz',
            '--adc_file_name', 'nonexistent.nii.gz',
        ]):
            original_join = os.path.join
            
            def mock_join_for_main(*args):
                if args and args[0] == '/app':
                    return original_join(tmp_dir, *args[1:])
                return original_join(*args)
            
            with patch.object(main.os.path, 'join', side_effect=mock_join_for_main):
                with patch.object(main.os.path, 'exists', return_value=False):
                    with pytest.raises(FileNotFoundError):
                        main.main()
