"""
Unit tests for DataModule class
"""

import pytest
import json
import tempfile
from pathlib import Path
import datetime
import shutil
import os

from armor_pipeline.data.dataset import DataModule
from armor_pipeline.data.parser import Annotation, Defect, DefectType, Point


class TestDataModule:
    """Test DataModule class functionality"""
    
    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotations for testing"""
        annotations = []
        
        # Create annotation with defect1
        defect1 = Defect(
            defect_type=DefectType.BBOX,
            name="defect1",
            x_min=10,
            y_min=20,
            x_max=30,
            y_max=40
        )
        
        annotation1 = Annotation(
            image_path=Path("/dummy/image1.bmp"),
            defects=[defect1]
        )
        
        # Create annotation with defect2
        defect2 = Defect(
            defect_type=DefectType.BBOX,
            name="defect2",
            x_min=50,
            y_min=60,
            x_max=70,
            y_max=80
        )
        
        annotation2 = Annotation(
            image_path=Path("/dummy/image2.bmp"),
            defects=[defect2]
        )
        
        annotations.extend([annotation1, annotation2])
        return annotations
    
    @pytest.fixture
    def mock_data_module(self, monkeypatch, sample_annotations):
        """Create a mock DataModule for testing"""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config directories
            data_path = temp_path / "data"
            mappings_path = data_path / "mappings"
            os.makedirs(mappings_path, exist_ok=True)
            
            # Create a mock DataModule
            data_module = DataModule(
                data_root=temp_path,
                batch_size=4,
                num_workers=1,
                img_size=1024,
                train_split=0.8,
                use_polygons=False,
                mapping_file=mappings_path / "class_mapping.json",
                allow_new_classes=True,
                load_existing_mapping=False,
                save_mapping=True
            )
            
            # Mock the all_annotations attribute
            data_module.all_annotations = sample_annotations
            
            # Mock the class_mapping attribute
            data_module.class_mapping = {
                "defect1": 1,
                "defect2": 2
            }
            
            yield data_module
            
            # Clean up
            shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_save_class_mapping(self, mock_data_module):
        """Test saving class mapping to a JSON file with metadata"""
        # Save class mapping
        output_path = mock_data_module.save_class_mapping()
        
        # Check that the file exists
        assert output_path.exists()
        
        # Load the saved mapping
        with open(output_path, 'r') as f:
            mapping_data = json.load(f)
        
        # Check structure
        assert "metadata" in mapping_data
        assert "class_mapping" in mapping_data
        assert "samples_per_class" in mapping_data
        
        # Check metadata
        metadata = mapping_data["metadata"]
        assert "creation_timestamp" in metadata
        assert "dataset_name" in metadata
        assert "total_samples" in metadata
        assert metadata["total_samples"] == 2  # Two sample annotations
        
        # Check class mapping
        class_mapping = mapping_data["class_mapping"]
        assert class_mapping == {"defect1": 1, "defect2": 2}
        
        # Check samples per class
        samples_per_class = mapping_data["samples_per_class"]
        assert samples_per_class == {"defect1": 1, "defect2": 1}
    
    def test_load_class_mapping(self, mock_data_module):
        """Test loading class mapping from a JSON file"""
        # First save a class mapping
        output_path = mock_data_module.save_class_mapping()
        
        # Create a new data module with load_existing_mapping=True
        new_data_module = DataModule(
            data_root=mock_data_module.project_config.base_path,
            mapping_file=output_path,
            load_existing_mapping=True
        )
        
        # Mock the all_annotations attribute (needed for setup)
        new_data_module.all_annotations = mock_data_module.all_annotations
        
        # Load the class mapping
        loaded_mapping = new_data_module.load_class_mapping()
        
        # Check that the loaded mapping matches the original
        assert loaded_mapping == mock_data_module.class_mapping
        assert new_data_module.class_mapping == mock_data_module.class_mapping
    
    def test_load_class_mapping_file_not_found(self, mock_data_module):
        """Test loading class mapping from a non-existent file"""
        # Create a non-existent file path
        non_existent_path = mock_data_module.project_config.data_path / "non_existent.json"
        
        # Try to load the class mapping
        with pytest.raises(FileNotFoundError):
            mock_data_module.load_class_mapping(non_existent_path)
    
    def test_load_class_mapping_invalid_json(self, mock_data_module):
        """Test loading class mapping from an invalid JSON file"""
        # Create an invalid JSON file
        invalid_path = mock_data_module.project_config.data_path / "invalid.json"
        with open(invalid_path, 'w') as f:
            f.write("invalid json content")
        
        # Try to load the class mapping
        with pytest.raises(json.JSONDecodeError):
            mock_data_module.load_class_mapping(invalid_path)
    
    def test_load_class_mapping_invalid_structure(self, mock_data_module):
        """Test loading class mapping from a file with invalid structure"""
        # Create a JSON file with invalid structure
        invalid_path = mock_data_module.project_config.data_path / "invalid_structure.json"
        with open(invalid_path, 'w') as f:
            json.dump({"invalid": "structure"}, f)
        
        # Try to load the class mapping
        with pytest.raises(KeyError):
            mock_data_module.load_class_mapping(invalid_path)
    
    def test_setup_with_load_existing_mapping(self, monkeypatch, mock_data_module):
        """Test setup with load_existing_mapping=True"""
        # First save a class mapping
        output_path = mock_data_module.save_class_mapping()
        
        # Create a new data module with load_existing_mapping=True
        new_data_module = DataModule(
            data_root=mock_data_module.project_config.base_path,
            mapping_file=output_path,
            load_existing_mapping=True
        )
        
        # Mock the parse_all_annotations function
        def mock_parse_all_annotations(*args, **kwargs):
            return mock_data_module.all_annotations
        
        # Mock the create_stratified_splits function
        def mock_create_stratified_splits(data, test_size, val_size):
            return data[:1], data[1:]
        
        # Mock the create_dataloaders function
        def mock_create_dataloaders(*args, **kwargs):
            return None, None, mock_data_module.class_mapping
        
        # Apply the monkeypatches
        monkeypatch.setattr("armor_pipeline.data.parser.parse_all_annotations", mock_parse_all_annotations)
        monkeypatch.setattr("armor_pipeline.data.dataset.create_stratified_splits", mock_create_stratified_splits)
        monkeypatch.setattr("armor_pipeline.data.dataset.create_dataloaders", mock_create_dataloaders)
        
        # Call setup
        new_data_module.setup()
        
        # Check that the class mapping was loaded
        assert new_data_module.class_mapping == mock_data_module.class_mapping
    
    def test_setup_with_save_mapping(self, monkeypatch, mock_data_module):
        """Test setup with save_mapping=True"""
        # Create a new data module with save_mapping=True
        new_data_module = DataModule(
            data_root=mock_data_module.project_config.base_path,
            save_mapping=True
        )
        
        # Mock the parse_all_annotations function
        def mock_parse_all_annotations(*args, **kwargs):
            return mock_data_module.all_annotations
        
        # Mock the create_stratified_splits function
        def mock_create_stratified_splits(data, test_size, val_size):
            return data[:1], data[1:]
        
        # Mock the create_dataloaders function
        def mock_create_dataloaders(*args, **kwargs):
            return None, None, mock_data_module.class_mapping
        
        # Apply the monkeypatches
        monkeypatch.setattr("armor_pipeline.data.parser.parse_all_annotations", mock_parse_all_annotations)
        monkeypatch.setattr("armor_pipeline.data.dataset.create_stratified_splits", mock_create_stratified_splits)
        monkeypatch.setattr("armor_pipeline.data.dataset.create_dataloaders", mock_create_dataloaders)
        
        # Call setup
        new_data_module.setup()
        
        # Check that the class mapping was saved
        assert new_data_module.mapping_file.exists()