"""
Unit tests for XML annotation parser
"""

import pytest
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import json
import os
import shutil
from armor_pipeline.data.parser import XMLAnnotationParser, DefectType, Defect, Point, parse_all_annotations


class TestXMLParser:
    """Test XML annotation parsing functionality"""

    @pytest.fixture
    def sample_xml(self):
        """Create sample XML content"""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <annotation>
            <filename>A/sample.bmp</filename>
            <size>
                <width>2048</width>
                <height>4096</height>
            </size>
            <object>
                <name>edge_tear</name>
                <type>polyline</type>
                <polyline>
                    <point x="100" y="200"/>
                    <point x="150" y="250"/>
                    <point x="200" y="300"/>
                </polyline>
            </object>
            <object>
                <name>bubble</name>
                <type>ellipse</type>
                <ellipse>
                    <center x="500" y="600"/>
                    <rx>50</rx>
                    <ry>40</ry>
                    <angle>30</angle>
                </ellipse>
            </object>
            <object>
                <name>surface_blob</name>
                <type>polygon</type>
                <polygon>
                    <point x="800" y="900"/>
                    <point x="850" y="920"/>
                    <point x="840" y="980"/>
                    <point x="790" y="950"/>
                </polygon>
            </object>
            <object>
                <name>macro_defect</name>
                <type>bbox</type>
                <bndbox>
                    <xmin>1000</xmin>
                    <ymin>1200</ymin>
                    <xmax>1200</xmax>
                    <ymax>1400</ymax>
                </bndbox>
            </object>
        </annotation>"""
        return xml_content

    def test_parse_polyline(self, sample_xml):
        """Test polyline parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(sample_xml)
            xml_path = Path(f.name)

        parser = XMLAnnotationParser(crop_to_upper_half=True)
        annotation = parser.parse_file(xml_path, Path("/dummy"))

        # Find polyline defect
        polyline_defects = [d for d in annotation.defects if d.defect_type == DefectType.POLYLINE]
        assert len(polyline_defects) == 1

        defect = polyline_defects[0]
        assert defect.name == "edge_tear"
        assert len(defect.points) == 3
        assert defect.points[0].x == 100
        assert defect.points[0].y == 200

        xml_path.unlink()

    def test_parse_ellipse(self, sample_xml):
        """Test ellipse parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(sample_xml)
            xml_path = Path(f.name)

        parser = XMLAnnotationParser(crop_to_upper_half=True)
        annotation = parser.parse_file(xml_path, Path("/dummy"))

        # Find ellipse defect
        ellipse_defects = [d for d in annotation.defects if d.defect_type == DefectType.ELLIPSE]
        assert len(ellipse_defects) == 1

        defect = ellipse_defects[0]
        assert defect.name == "bubble"
        assert defect.center.x == 500
        assert defect.center.y == 600
        assert defect.rx == 50
        assert defect.ry == 40
        assert defect.angle == 30

        xml_path.unlink()

    def test_bbox_conversion(self):
        """Test conversion of different defect types to bounding boxes"""
        # Test polyline to bbox
        polyline = Defect(
            defect_type=DefectType.POLYLINE,
            name="scratch",
            points=[Point(100, 200), Point(200, 300), Point(150, 400)]
        )
        bbox = polyline.to_bbox()
        assert bbox == (100, 200, 200, 400)

        # Test ellipse to bbox
        ellipse = Defect(
            defect_type=DefectType.ELLIPSE,
            name="bubble",
            center=Point(500, 500),
            rx=100,
            ry=50,
            angle=0
        )
        bbox = ellipse.to_bbox()
        assert bbox == (400, 450, 600, 550)

    def test_crop_to_upper_half(self, sample_xml):
        """Test that defects in lower half are filtered out"""
        # Add defect in lower half
        xml_with_lower = sample_xml.replace(
            "</annotation>",
            """<object>
                <name>lower_defect</name>
                <type>bbox</type>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>3000</ymin>
                    <xmax>200</xmax>
                    <ymax>3100</ymax>
                </bndbox>
            </object>
            </annotation>"""
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_with_lower)
            xml_path = Path(f.name)

        parser = XMLAnnotationParser(crop_to_upper_half=True)
        annotation = parser.parse_file(xml_path, Path("/dummy"))

        # Check that lower defect is not included
        defect_names = [d.name for d in annotation.defects]
        assert "lower_defect" not in defect_names

        xml_path.unlink()
        
    def test_extract_image_filename(self):
        """Test image filename extraction from different XML structures"""
        parser = XMLAnnotationParser()
        
        # Case 1: Filename in XML using root.find(".//filename")
        xml_with_filename = """<?xml version="1.0" encoding="UTF-8"?>
        <annotation>
            <filename>path/to/image.bmp</filename>
        </annotation>"""
        
        root = ET.fromstring(xml_with_filename)
        xml_path = Path("/dummy/TAM17-A/annotation.xml")
        filename = parser._extract_image_filename(root, xml_path)
        assert filename == "image.bmp"
        
        # Case 2: Extract from XML structure patterns like TAM17-A folders
        xml_without_filename = """<?xml version="1.0" encoding="UTF-8"?>
        <annotation>
        </annotation>"""
        
        root = ET.fromstring(xml_without_filename)
        xml_path = Path("/dummy/TAM17-A/annotation.xml")
        filename = parser._extract_image_filename(root, xml_path)
        assert filename == "A.bmp"
        
        # Case 3: Look for image references in XML using root.findall(".//image")
        xml_with_image_ref = """<?xml version="1.0" encoding="UTF-8"?>
        <annotation>
            <image src="referenced_image.bmp"/>
        </annotation>"""
        
        root = ET.fromstring(xml_with_image_ref)
        xml_path = Path("/dummy/TAM17-A/annotation.xml")
        filename = parser._extract_image_filename(root, xml_path)
        assert filename == "referenced_image.bmp"
        
        # Case 4: Fallback to pattern-based naming using folder_name.bmp
        xml_without_any_ref = """<?xml version="1.0" encoding="UTF-8"?>
        <annotation>
        </annotation>"""
        
        root = ET.fromstring(xml_without_any_ref)
        xml_path = Path("/dummy/SomeFolder/annotation.xml")
        filename = parser._extract_image_filename(root, xml_path)
        assert filename == "SomeFolder.bmp"
        
    def test_find_local_image_path(self):
        """Test finding local image path using various strategies"""
        parser = XMLAnnotationParser()
        
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Test direct path lookup
            direct_image_path = temp_path / "direct.bmp"
            with open(direct_image_path, 'w') as f:
                f.write("dummy image content")
                
            result = parser.find_local_image_path("direct.bmp", temp_path)
            assert result == direct_image_path
            
            # 2. Test manifest.json filename mappings
            manifest_data = {
                "files": {
                    "mapped.bmp": "actual_mapped_file.bmp"
                }
            }
            
            manifest_path = temp_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
                
            mapped_image_path = temp_path / "actual_mapped_file.bmp"
            with open(mapped_image_path, 'w') as f:
                f.write("dummy mapped image content")
                
            result = parser.find_local_image_path("mapped.bmp", temp_path)
            assert result == mapped_image_path
            
            # 3. Test pattern matching with different extensions
            # Create files with different extensions
            stem_name = "multi_ext"
            for ext in [".BMP", ".jpg", ".png"]:
                with open(temp_path / f"{stem_name}{ext}", 'w') as f:
                    f.write(f"dummy {ext} content")
            
            # Test finding with original extension not present but alternatives exist
            result = parser.find_local_image_path(f"{stem_name}.bmp", temp_path)
            assert result == temp_path / f"{stem_name}.BMP"
            
            # 4. Test recursive search fallback
            # Create nested directory structure
            nested_dir = temp_path / "nested" / "deep"
            os.makedirs(nested_dir, exist_ok=True)
            
            nested_image_path = nested_dir / "nested_image.bmp"
            with open(nested_image_path, 'w') as f:
                f.write("dummy nested image content")
                
            result = parser.find_local_image_path("nested_image.bmp", temp_path)
            assert result == nested_image_path
            
            # 5. Test handling of not found cases
            result = parser.find_local_image_path("nonexistent.bmp", temp_path)
            assert result is None
            
    def test_load_from_manifest(self):
        """Test loading parser from manifest.json file"""
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a manifest file with filename mappings
            manifest_data = {
                "files": {
                    "original.bmp": "mapped_file.bmp",
                    "another.bmp": "another_mapped.bmp"
                },
                "metadata": {
                    "source": "S3 extraction",
                    "date": "2025-07-29"
                }
            }
            
            manifest_path = temp_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
                
            # Create the mapped image files
            mapped_image_path = temp_path / "mapped_file.bmp"
            with open(mapped_image_path, 'w') as f:
                f.write("dummy mapped image content")
                
            another_mapped_path = temp_path / "another_mapped.bmp"
            with open(another_mapped_path, 'w') as f:
                f.write("another dummy mapped image content")
                
            # Create a sample XML file
            xml_content = """<?xml version="1.0" encoding="UTF-8"?>
            <annotation>
                <filename>original.bmp</filename>
                <object>
                    <name>test_defect</name>
                    <type>polyline</type>
                    <polyline>
                        <point x="100" y="200"/>
                        <point x="150" y="250"/>
                    </polyline>
                </object>
            </annotation>"""
            
            xml_path = temp_path / "annotation.xml"
            with open(xml_path, 'w') as f:
                f.write(xml_content)
                
            # Test loading parser from manifest
            parser = XMLAnnotationParser.load_from_manifest(manifest_path)
            
            # Verify parser attributes
            assert parser.image_dir == temp_path
            assert parser.manifest_path == manifest_path
            assert parser.manifest == manifest_data
            assert parser.filename_mappings == manifest_data["files"]
            
            # Test parsing XML file using the loaded parser
            annotation = parser.parse_file(xml_path)
            
            # Verify that the correct image path was used (mapped path)
            assert annotation.image_path == mapped_image_path
            
            # Verify that defects were parsed correctly
            assert len(annotation.defects) == 1
            assert annotation.defects[0].name == "test_defect"
            assert annotation.defects[0].defect_type == DefectType.POLYLINE
            
    def test_load_from_manifest_error_handling(self):
        """Test error handling when loading parser from manifest.json file"""
        # Test with non-existent manifest file
        with pytest.raises(FileNotFoundError):
            XMLAnnotationParser.load_from_manifest(Path("/nonexistent/manifest.json"))
            
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_path = Path(f.name)
            
        try:
            with pytest.raises(json.JSONDecodeError):
                XMLAnnotationParser.load_from_manifest(invalid_path)
        finally:
            invalid_path.unlink()
            
    def test_parse_file_without_image_dir(self):
        """Test parsing file without providing image_dir"""
        parser = XMLAnnotationParser()
        
        # Create a temporary XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <annotation>
            <filename>test.bmp</filename>
        </annotation>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            xml_path = Path(f.name)
            
        try:
            # Test parsing without image_dir and without setting parser.image_dir
            with pytest.raises(ValueError, match="No image directory provided"):
                parser.parse_file(xml_path)
                
            # Test parsing after setting parser.image_dir
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create test image
                test_image_path = temp_path / "test.bmp"
                with open(test_image_path, 'w') as f:
                    f.write("dummy image content")
                    
                # Set image_dir on parser instance
                parser.image_dir = temp_path
                
                # Parse file without providing image_dir parameter
                annotation = parser.parse_file(xml_path)
                
                # Verify that the correct image path was used
                assert annotation.image_path == test_image_path
        finally:
            xml_path.unlink()
    def test_parse_all_annotations_without_manifest(self):
        """Test parse_all_annotations function without manifest path"""
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create annotations directory
            annotations_dir = temp_path / "annotations"
            os.makedirs(annotations_dir, exist_ok=True)
            
            # Create images directory
            images_dir = temp_path / "images"
            os.makedirs(images_dir, exist_ok=True)
            
            # Create sample XML files
            xml_content1 = """<?xml version="1.0" encoding="UTF-8"?>
            <annotation>
                <filename>image1.bmp</filename>
                <object>
                    <name>defect1</name>
                    <type>polyline</type>
                    <polyline>
                        <point x="100" y="200"/>
                        <point x="150" y="250"/>
                    </polyline>
                </object>
            </annotation>"""
            
            xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
            <annotation>
                <filename>image2.bmp</filename>
                <object>
                    <name>defect2</name>
                    <type>ellipse</type>
                    <ellipse>
                        <center x="500" y="600"/>
                        <rx>50</rx>
                        <ry>40</ry>
                        <angle>30</angle>
                    </ellipse>
                </object>
            </annotation>"""
            
            # Write XML files
            xml_path1 = annotations_dir / "annotation1.xml"
            with open(xml_path1, 'w') as f:
                f.write(xml_content1)
                
            xml_path2 = annotations_dir / "annotation2.xml"
            with open(xml_path2, 'w') as f:
                f.write(xml_content2)
                
            # Create image files
            image_path1 = images_dir / "image1.bmp"
            with open(image_path1, 'w') as f:
                f.write("dummy image content")
                
            image_path2 = images_dir / "image2.bmp"
            with open(image_path2, 'w') as f:
                f.write("dummy image content")
                
            # Test parse_all_annotations without manifest_path
            annotations = parse_all_annotations(
                annotations_dir=annotations_dir,
                images_dir=images_dir,
                crop_to_upper_half=True
            )
            
            # Verify results
            assert len(annotations) == 2
            
            # Check first annotation
            assert annotations[0].image_path == image_path1
            assert len(annotations[0].defects) == 1
            assert annotations[0].defects[0].name == "defect1"
            assert annotations[0].defects[0].defect_type == DefectType.POLYLINE
            
            # Check second annotation
            assert annotations[1].image_path == image_path2
            assert len(annotations[1].defects) == 1
            assert annotations[1].defects[0].name == "defect2"
            assert annotations[1].defects[0].defect_type == DefectType.ELLIPSE
            
    def test_parse_all_annotations_with_manifest(self):
        """Test parse_all_annotations function with manifest path"""
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create annotations directory
            annotations_dir = temp_path / "annotations"
            os.makedirs(annotations_dir, exist_ok=True)
            
            # Create extracted directory (where manifest and images are)
            extracted_dir = temp_path / "extracted"
            os.makedirs(extracted_dir, exist_ok=True)
            
            # Create manifest file with filename mappings
            manifest_data = {
                "files": {
                    "image1.bmp": "mapped_image1.bmp",
                    "image2.bmp": "mapped_image2.bmp"
                },
                "metadata": {
                    "source": "S3 extraction",
                    "date": "2025-07-29"
                }
            }
            
            manifest_path = extracted_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
                
            # Create mapped image files
            mapped_image_path1 = extracted_dir / "mapped_image1.bmp"
            with open(mapped_image_path1, 'w') as f:
                f.write("dummy mapped image content")
                
            mapped_image_path2 = extracted_dir / "mapped_image2.bmp"
            with open(mapped_image_path2, 'w') as f:
                f.write("dummy mapped image content")
                
            # Create sample XML files
            xml_content1 = """<?xml version="1.0" encoding="UTF-8"?>
            <annotation>
                <filename>image1.bmp</filename>
                <object>
                    <name>defect1</name>
                    <type>polyline</type>
                    <polyline>
                        <point x="100" y="200"/>
                        <point x="150" y="250"/>
                    </polyline>
                </object>
            </annotation>"""
            
            xml_content2 = """<?xml version="1.0" encoding="UTF-8"?>
            <annotation>
                <filename>image2.bmp</filename>
                <object>
                    <name>defect2</name>
                    <type>ellipse</type>
                    <ellipse>
                        <center x="500" y="600"/>
                        <rx>50</rx>
                        <ry>40</ry>
                        <angle>30</angle>
                    </ellipse>
                </object>
            </annotation>"""
            
            # Write XML files
            xml_path1 = annotations_dir / "annotation1.xml"
            with open(xml_path1, 'w') as f:
                f.write(xml_content1)
                
            xml_path2 = annotations_dir / "annotation2.xml"
            with open(xml_path2, 'w') as f:
                f.write(xml_content2)
                
            # Test parse_all_annotations with manifest_path
            annotations = parse_all_annotations(
                annotations_dir=annotations_dir,
                crop_to_upper_half=True,
                manifest_path=manifest_path
            )
            
            # Verify results
            assert len(annotations) == 2
            
            # Check first annotation - should use mapped image path
            assert annotations[0].image_path == mapped_image_path1
            assert len(annotations[0].defects) == 1
            assert annotations[0].defects[0].name == "defect1"
            assert annotations[0].defects[0].defect_type == DefectType.POLYLINE
            
            # Check second annotation - should use mapped image path
            assert annotations[1].image_path == mapped_image_path2
            assert len(annotations[1].defects) == 1
            assert annotations[1].defects[0].name == "defect2"
            assert annotations[1].defects[0].defect_type == DefectType.ELLIPSE
            
    def test_parse_all_annotations_error_cases(self):
        """Test error cases for parse_all_annotations function"""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create annotations directory
            annotations_dir = temp_path / "annotations"
            os.makedirs(annotations_dir, exist_ok=True)
            
            # Test case: Neither images_dir nor manifest_path provided
            with pytest.raises(ValueError, match="images_dir must be provided when manifest_path is not provided"):
                parse_all_annotations(
                    annotations_dir=annotations_dir,
                    crop_to_upper_half=True
                )
                
            # Test case: Non-existent manifest file
            with pytest.raises(FileNotFoundError):
                parse_all_annotations(
                    annotations_dir=annotations_dir,
                    crop_to_upper_half=True,
                    manifest_path=temp_path / "nonexistent_manifest.json"
                )
                
            # Test case: Invalid manifest file
            invalid_manifest_path = temp_path / "invalid_manifest.json"
            with open(invalid_manifest_path, 'w') as f:
                f.write("invalid json content")
                
            with pytest.raises(json.JSONDecodeError):
                parse_all_annotations(
                    annotations_dir=annotations_dir,
                    crop_to_upper_half=True,
                    manifest_path=invalid_manifest_path
                )