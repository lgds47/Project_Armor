"""
Unit tests for class mapping functionality in ContactLensDataset
"""

import pytest
import json
import tempfile
from pathlib import Path
import shutil

from armor_pipeline.data.dataset import ContactLensDataset
from armor_pipeline.data.parser import Annotation, Defect, DefectType, Point


class TestClassMapping:
    """Test class mapping functionality in ContactLensDataset"""
    
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
    def sample_annotations_with_new_class(self):
        """Create sample annotations with a new class for testing"""
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
        
        # Create annotation with new_defect
        new_defect = Defect(
            defect_type=DefectType.BBOX,
            name="new_defect",
            x_min=90,
            y_min=100,
            x_max=110,
            y_max=120
        )
        
        annotation3 = Annotation(
            image_path=Path("/dummy/image3.bmp"),
            defects=[new_defect]
        )
        
        annotations.extend([annotation1, annotation2, annotation3])
        return annotations
    
    def test_build_class_mapping_without_file(self, sample_annotations):
        """Test building class mapping without a mapping file"""
        dataset = ContactLensDataset(
            annotations=sample_annotations,
            transform=None,
            mode="test",
            use_polygons=False,
            class_mapping=None,
            mapping_file=None
        )
        
        # Check that the class mapping was created correctly
        assert len(dataset.class_mapping) == 2
        assert "defect1" in dataset.class_mapping
        assert "defect2" in dataset.class_mapping
        assert dataset.class_mapping["defect1"] == 1
        assert dataset.class_mapping["defect2"] == 2
    
    def test_save_and_load_class_mapping(self, sample_annotations):
        """Test saving and loading class mapping to/from a file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mapping_file = temp_path / "class_mapping.json"
            
            # Create dataset and save mapping
            dataset1 = ContactLensDataset(
                annotations=sample_annotations,
                transform=None,
                mode="test",
                use_polygons=False,
                class_mapping=None,
                mapping_file=mapping_file
            )
            
            # Check that the mapping file was created
            assert mapping_file.exists()
            
            # Load the mapping from the file
            with open(mapping_file, 'r') as f:
                saved_mapping = json.load(f)
            
            # Check that the saved mapping is correct
            assert len(saved_mapping) == 2
            assert "defect1" in saved_mapping
            assert "defect2" in saved_mapping
            assert saved_mapping["defect1"] == 1
            assert saved_mapping["defect2"] == 2
            
            # Create a new dataset with the same mapping file
            dataset2 = ContactLensDataset(
                annotations=sample_annotations,
                transform=None,
                mode="test",
                use_polygons=False,
                class_mapping=None,
                mapping_file=mapping_file
            )
            
            # Check that the loaded mapping is the same as the saved mapping
            assert dataset2.class_mapping == dataset1.class_mapping
    
    def test_validate_new_classes(self, sample_annotations, sample_annotations_with_new_class):
        """Test validation of new classes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mapping_file = temp_path / "class_mapping.json"
            
            # Create dataset with initial annotations and save mapping
            dataset1 = ContactLensDataset(
                annotations=sample_annotations,
                transform=None,
                mode="test",
                use_polygons=False,
                class_mapping=None,
                mapping_file=mapping_file
            )
            
            # Try to create a dataset with new classes but allow_new_classes=False
            with pytest.raises(ValueError, match="Found .* new classes not in existing mapping"):
                dataset2 = ContactLensDataset(
                    annotations=sample_annotations_with_new_class,
                    transform=None,
                    mode="test",
                    use_polygons=False,
                    class_mapping=None,
                    mapping_file=mapping_file,
                    allow_new_classes=False
                )
    
    def test_allow_new_classes(self, sample_annotations, sample_annotations_with_new_class):
        """Test allowing new classes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mapping_file = temp_path / "class_mapping.json"
            
            # Create dataset with initial annotations and save mapping
            dataset1 = ContactLensDataset(
                annotations=sample_annotations,
                transform=None,
                mode="test",
                use_polygons=False,
                class_mapping=None,
                mapping_file=mapping_file
            )
            
            # Create a dataset with new classes and allow_new_classes=True
            dataset2 = ContactLensDataset(
                annotations=sample_annotations_with_new_class,
                transform=None,
                mode="test",
                use_polygons=False,
                class_mapping=None,
                mapping_file=mapping_file,
                allow_new_classes=True
            )
            
            # Check that the new class was added to the mapping
            assert len(dataset2.class_mapping) == 3
            assert "defect1" in dataset2.class_mapping
            assert "defect2" in dataset2.class_mapping
            assert "new_defect" in dataset2.class_mapping
            assert dataset2.class_mapping["defect1"] == 1
            assert dataset2.class_mapping["defect2"] == 2
            assert dataset2.class_mapping["new_defect"] == 3
            
            # Load the updated mapping from the file
            with open(mapping_file, 'r') as f:
                updated_mapping = json.load(f)
            
            # Check that the updated mapping was saved correctly
            assert len(updated_mapping) == 3
            assert "defect1" in updated_mapping
            assert "defect2" in updated_mapping
            assert "new_defect" in updated_mapping
            assert updated_mapping["defect1"] == 1
            assert updated_mapping["defect2"] == 2
            assert updated_mapping["new_defect"] == 3
    
    def test_deterministic_sorting(self):
        """Test that class mapping uses deterministic sorting"""
        # Create annotations with defects in different order
        annotations1 = []
        
        defect1 = Defect(
            defect_type=DefectType.BBOX,
            name="zebra",
            x_min=10,
            y_min=20,
            x_max=30,
            y_max=40
        )
        
        defect2 = Defect(
            defect_type=DefectType.BBOX,
            name="apple",
            x_min=50,
            y_min=60,
            x_max=70,
            y_max=80
        )
        
        annotation1 = Annotation(
            image_path=Path("/dummy/image1.bmp"),
            defects=[defect1, defect2]
        )
        
        annotations1.append(annotation1)
        
        # Create annotations with defects in reverse order
        annotations2 = []
        
        defect3 = Defect(
            defect_type=DefectType.BBOX,
            name="apple",
            x_min=50,
            y_min=60,
            x_max=70,
            y_max=80
        )
        
        defect4 = Defect(
            defect_type=DefectType.BBOX,
            name="zebra",
            x_min=10,
            y_min=20,
            x_max=30,
            y_max=40
        )
        
        annotation2 = Annotation(
            image_path=Path("/dummy/image2.bmp"),
            defects=[defect3, defect4]
        )
        
        annotations2.append(annotation2)
        
        # Create datasets with different annotation orders
        dataset1 = ContactLensDataset(
            annotations=annotations1,
            transform=None,
            mode="test",
            use_polygons=False,
            class_mapping=None,
            mapping_file=None
        )
        
        dataset2 = ContactLensDataset(
            annotations=annotations2,
            transform=None,
            mode="test",
            use_polygons=False,
            class_mapping=None,
            mapping_file=None
        )
        
        # Check that the class mappings are the same regardless of order
        assert dataset1.class_mapping == dataset2.class_mapping
        assert dataset1.class_mapping["apple"] == 1  # "apple" should come first alphabetically
        assert dataset1.class_mapping["zebra"] == 2  # "zebra" should come second alphabetically