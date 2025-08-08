"""
XML Parser for J&J Contact Lens Defect Annotations
Parses custom XML format into unified schema for training
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import numpy as np
from enum import Enum
import json
import logging
from armor_pipeline.data.s3_image_loader import S3ImageLoader
from armor_pipeline.data.defect_taxonomy import JJDefectTaxonomy


class DefectType(Enum):
    POLYLINE = "polyline"  # Edge tears/scratches
    POLYGON = "polygon"    # Surface blobs
    ELLIPSE = "ellipse"    # Bubbles, rings
    BBOX = "bbox"          # Macro defects


@dataclass
class Point:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Defect:
    """Unified defect representation"""
    defect_type: DefectType
    name: str
    points: List[Point] = field(default_factory=list)

    # For ellipse
    center: Optional[Point] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    angle: Optional[float] = None

    # For bbox
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert any defect type to bounding box for detection models"""
        if self.defect_type == DefectType.BBOX:
            return (self.x_min, self.y_min, self.x_max, self.y_max)

        elif self.defect_type == DefectType.ELLIPSE:
            # Convert ellipse to bbox
            cx, cy = self.center.x, self.center.y
            # Account for rotation
            cos_a = np.cos(np.radians(self.angle or 0))
            sin_a = np.sin(np.radians(self.angle or 0))

            # Calculate rotated bbox
            dx = np.sqrt((self.rx * cos_a) ** 2 + (self.ry * sin_a) ** 2)
            dy = np.sqrt((self.rx * sin_a) ** 2 + (self.ry * cos_a) ** 2)

            return (cx - dx, cy - dy, cx + dx, cy + dy)

        elif self.defect_type in [DefectType.POLYGON, DefectType.POLYLINE]:
            # Get bbox from points
            xs = [p.x for p in self.points]
            ys = [p.y for p in self.points]
            return (min(xs), min(ys), max(xs), max(ys))

        raise ValueError(f"Unknown defect type: {self.defect_type}")

    def to_polygon(self) -> List[Tuple[float, float]]:
        """Convert to polygon points for segmentation models"""
        if self.defect_type in [DefectType.POLYGON, DefectType.POLYLINE]:
            return [p.to_tuple() for p in self.points]

        elif self.defect_type == DefectType.ELLIPSE:
            # Sample ellipse points
            n_points = 32
            angles = np.linspace(0, 2 * np.pi, n_points)
            points = []

            for theta in angles:
                # Parametric ellipse equation with rotation
                x = self.rx * np.cos(theta)
                y = self.ry * np.sin(theta)

                # Rotate
                angle_rad = np.radians(self.angle or 0)
                x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

                # Translate to center
                points.append((self.center.x + x_rot, self.center.y + y_rot))

            return points

        elif self.defect_type == DefectType.BBOX:
            # Convert bbox to polygon
            return [
                (self.x_min, self.y_min),
                (self.x_max, self.y_min),
                (self.x_max, self.y_max),
                (self.x_min, self.y_max)
            ]

        raise ValueError(f"Cannot convert {self.defect_type} to polygon")


@dataclass
class Annotation:
    """Container for all defects in one image"""
    image_path: Path
    defects: List[Defect] = field(default_factory=list)
    width: int = 2048  # After cropping upper half
    height: int = 2048


class XMLAnnotationParser:
    """Parser for J&J custom XML annotation format"""

    SUPPORTED_TYPES = {"polyline", "polygon", "ellipse", "bbox", "bounding_box"}
    
    # Class logger
    logger = logging.getLogger("armor_pipeline.parser")

    @classmethod
    def load_from_manifest(cls, manifest_path: Path, crop_to_upper_half: bool = True) -> 'XMLAnnotationParser':
        """
        Load parser from manifest.json file.
        
        Args:
            manifest_path: Path to the manifest.json file
            crop_to_upper_half: Whether to crop defects to upper half of image
            
        Returns:
            Configured XMLAnnotationParser instance
        
        Raises:
            FileNotFoundError: If manifest file doesn't exist
            json.JSONDecodeError: If manifest file is not valid JSON
            KeyError: If manifest file doesn't have expected structure
        """
        # Check if manifest file exists
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            
        # Read manifest file
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            cls.logger.error(f"Failed to parse manifest file {manifest_path}: {e}")
            raise
            
        # Create parser instance
        parser = cls(crop_to_upper_half=crop_to_upper_half)
        
        # Set up image directory path based on manifest location
        # The image directory is the same directory where the manifest is located
        parser.image_dir = manifest_path.parent
        
        # Store the manifest path and content
        parser.manifest_path = manifest_path
        parser.manifest = manifest
        
        # Extract filename mappings if available
        parser.filename_mappings = {}
        if "files" in manifest:
            parser.filename_mappings = manifest["files"]
            cls.logger.info(f"Loaded {len(parser.filename_mappings)} filename mappings from manifest")
        
        # Log successful loading
        cls.logger.info(f"Loaded parser from manifest: {manifest_path}")
        
        return parser

    def __init__(self, crop_to_upper_half: bool = True):
        self.crop_to_upper_half = crop_to_upper_half
        self.original_height = 4096
        self.cropped_height = 2048
        self.image_dir = None
        self.manifest_path = None
        self.manifest = None
        self.filename_mappings = {}

    def parse_file(self, xml_path: Path, image_dir: Optional[Path] = None) -> Annotation:
        """
        Parse single XML annotation file
        
        Args:
            xml_path: Path to the XML annotation file
            image_dir: Directory containing image files. If None, uses self.image_dir if available
            
        Returns:
            Annotation object
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Use provided image_dir or fall back to self.image_dir if available
        effective_image_dir = image_dir if image_dir is not None else self.image_dir
        
        # If no image directory is available, raise an error
        if effective_image_dir is None:
            raise ValueError("No image directory provided. Either provide image_dir parameter or use load_from_manifest()")
        
        # Extract image filename from XML or infer from structure
        image_filename = self._extract_image_filename(root, xml_path)
        
        # Try to find the actual image file using various strategies
        found_path = self.find_local_image_path(image_filename, effective_image_dir)
        
        # Log warning if image not found
        if found_path is None:
            self.logger.warning(f"Image not found: {image_filename} in directory {effective_image_dir}")
            
        # Fall back to simple path joining if not found
        image_path = found_path if found_path else effective_image_dir / image_filename

        annotation = Annotation(image_path=image_path)

        # Parse all objects
        for obj in root.findall(".//object"):
            defect = self._parse_object(obj)
            if defect and self._is_in_upper_half(defect):
                annotation.defects.append(defect)

        return annotation

    def _extract_image_filename(self, root: ET.Element, xml_path: Path) -> str:
        """Extract image filename from XML or infer from path"""
        # 1. First try to find filename in XML using root.find(".//filename")
        filename_elem = root.find(".//filename")
        if filename_elem is not None and filename_elem.text:
            return Path(filename_elem.text).name
            
        # 2. If not found, extract from XML structure patterns like TAM17-A folders
        folder_name = xml_path.parent.name  # e.g., "TAM17-A"
        if "-" in folder_name:
            letter = folder_name.split('-')[-1]  # Extract "A" from "TAM17-A"
            # Check if letter is a single character (typical pattern)
            if len(letter) == 1:
                filename = f"{letter}.bmp"
                return filename
                
        # 3. Look for image references in XML using root.findall(".//image")
        image_elems = root.findall(".//image")
        if image_elems and len(image_elems) > 0:
            for img in image_elems:
                # Try to get filename from src, href or other common attributes
                for attr in ['src', 'href', 'source', 'file']:
                    if attr in img.attrib:
                        return Path(img.attrib[attr]).name
                        
        # 4. Fallback to pattern-based naming using folder_name.bmp
        # 5. Return just the basename using Path(filename).name
        return Path(f"{folder_name}.bmp").name
        
    def find_local_image_path(self, image_filename: str, image_dir: Path) -> Optional[Path]:
        """
        Find the actual path to an image file using various strategies.
        
        Args:
            image_filename: The filename of the image to find
            image_dir: The directory to search in
            
        Returns:
            Path object if found, None otherwise
        """
        # 1. Try direct path lookup first
        direct_path = image_dir / image_filename
        if direct_path.exists():
            return direct_path
            
        # 2. Check instance filename mappings from manifest if available
        if self.filename_mappings and image_filename in self.filename_mappings:
            mapped_path = image_dir / self.filename_mappings[image_filename]
            if mapped_path.exists():
                return mapped_path
            
        # 3. Check manifest.json if it exists for filename mappings
        manifest_path = image_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    
                # Check if the manifest contains a mapping for this filename
                if "files" in manifest and image_filename in manifest["files"]:
                    mapped_path = image_dir / manifest["files"][image_filename]
                    if mapped_path.exists():
                        return mapped_path
            except (json.JSONDecodeError, KeyError, TypeError):
                # Continue to next strategy if manifest parsing fails
                pass
                
        # 3. Try pattern matching with different extensions
        filename_stem = Path(image_filename).stem
        for ext in [".bmp", ".BMP", ".jpg", ".png"]:
            pattern_path = image_dir / f"{filename_stem}{ext}"
            if pattern_path.exists():
                return pattern_path
                
        # 4. Search recursively using rglob as fallback
        # First try to match the exact filename
        for found_path in image_dir.rglob(image_filename):
            if found_path.is_file():
                return found_path
                
        # Then try with just the stem and any extension
        for found_path in image_dir.rglob(f"{filename_stem}.*"):
            if found_path.is_file() and found_path.suffix.lower() in [".bmp", ".jpg", ".png"]:
                return found_path
                
        # 5. Last resort: look for any .bmp file if we're expecting a .bmp
        if Path(image_filename).suffix.lower() == ".bmp":
            for found_path in image_dir.rglob("*.bmp"):
                if found_path.is_file():
                    return found_path
                    
        # Not found
        return None

    def _parse_object(self, obj_elem: ET.Element) -> Optional[Defect]:
        """Parse single object element"""
        # Get object type and name
        type_elem = obj_elem.find("type")
        
        # Try to find name in either <name> or <n> tag
        name_elem = obj_elem.find("name")
        if name_elem is None:
            # Try alternative tag <n> used in some XML formats
            name_elem = obj_elem.find("n")

        if type_elem is None:
            return None

        obj_type = type_elem.text.lower()
        raw_name = name_elem.text if name_elem is not None else "unknown"
        # Normalize defect name using JJDefectTaxonomy
        obj_name = JJDefectTaxonomy.normalize_defect_name(raw_name)

        # Skip unsupported types
        if obj_type not in self.SUPPORTED_TYPES:
            return None

        # Parse based on type
        if obj_type == "polyline":
            return self._parse_polyline(obj_elem, obj_name)
        elif obj_type == "polygon":
            return self._parse_polygon(obj_elem, obj_name)
        elif obj_type == "ellipse":
            return self._parse_ellipse(obj_elem, obj_name)
        elif obj_type in ["bbox", "bounding_box"]:
            return self._parse_bbox(obj_elem, obj_name)

        return None

    def _parse_polyline(self, obj_elem: ET.Element, name: str) -> Optional[Defect]:
        """Parse polyline (edge tears/scratches)"""
        points = []

        for point_elem in obj_elem.findall(".//point"):
            x = float(point_elem.get("x", 0))
            y = float(point_elem.get("y", 0))
            if self.crop_to_upper_half:
                y = min(y, self.cropped_height)
            points.append(Point(x, y))

        if not points:
            return None

        return Defect(
            defect_type=DefectType.POLYLINE,
            name=name,
            points=points
        )

    def _parse_polygon(self, obj_elem: ET.Element, name: str) -> Optional[Defect]:
        """Parse polygon (surface blobs)"""
        points = []

        for point_elem in obj_elem.findall(".//point"):
            x = float(point_elem.get("x", 0))
            y = float(point_elem.get("y", 0))
            if self.crop_to_upper_half:
                y = min(y, self.cropped_height)
            points.append(Point(x, y))

        if len(points) < 3:  # Need at least 3 points for polygon
            return None

        return Defect(
            defect_type=DefectType.POLYGON,
            name=name,
            points=points
        )

    def _parse_ellipse(self, obj_elem: ET.Element, name: str) -> Optional[Defect]:
        """Parse ellipse (bubbles, rings)"""
        center_elem = obj_elem.find(".//center")
        if center_elem is None:
            return None

        cx = float(center_elem.get("x", 0))
        cy = float(center_elem.get("y", 0))

        if self.crop_to_upper_half:
            cy = min(cy, self.cropped_height)

        rx = float(obj_elem.find(".//rx").text) if obj_elem.find(".//rx") is not None else 0
        ry = float(obj_elem.find(".//ry").text) if obj_elem.find(".//ry") is not None else 0
        angle = float(obj_elem.find(".//angle").text) if obj_elem.find(".//angle") is not None else 0

        return Defect(
            defect_type=DefectType.ELLIPSE,
            name=name,
            center=Point(cx, cy),
            rx=rx,
            ry=ry,
            angle=angle
        )

    def _parse_bbox(self, obj_elem: ET.Element, name: str) -> Optional[Defect]:
        """Parse bounding box (macro defects)"""
        bbox_elem = obj_elem.find(".//bndbox")
        if bbox_elem is None:
            return None

        xmin = float(bbox_elem.find("xmin").text)
        ymin = float(bbox_elem.find("ymin").text)
        xmax = float(bbox_elem.find("xmax").text)
        ymax = float(bbox_elem.find("ymax").text)

        if self.crop_to_upper_half:
            ymin = min(ymin, self.cropped_height)
            ymax = min(ymax, self.cropped_height)

        return Defect(
            defect_type=DefectType.BBOX,
            name=name,
            x_min=xmin,
            y_min=ymin,
            x_max=xmax,
            y_max=ymax
        )

    def _is_in_upper_half(self, defect: Defect) -> bool:
        """Check if defect is in upper half of image"""
        if not self.crop_to_upper_half:
            return True

        bbox = defect.to_bbox()
        return bbox[1] < self.cropped_height  # Check if top of bbox is in upper half
        
    def parse_points(self, points_str: str) -> List[Point]:
        """Parse points string into list of Point objects"""
        points = []
        # Format is typically "x1,y1 x2,y2 x3,y3 ..."
        for point_pair in points_str.strip().split():
            try:
                x, y = map(float, point_pair.split(','))
                points.append(Point(x, y))
            except ValueError:
                continue  # Skip invalid points
        return points
        
    def _polyline_length(self, points: List[Point]) -> float:
        """Calculate the length of a polyline"""
        if len(points) < 2:
            return 0.0
            
        length = 0.0
        for i in range(len(points) - 1):
            # Calculate Euclidean distance between consecutive points
            dx = points[i+1].x - points[i].x
            dy = points[i+1].y - points[i].y
            length += np.sqrt(dx*dx + dy*dy)
            
        return length
        
    def parse_polyline_measurements(self, xml_path: Path) -> Dict[str, float]:
        """Extract polyline depth measurements for edge defects"""
        # Parse companion polylines for depth
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        polylines = {}
        for polyline in root.findall('.//polyline'):
            label = polyline.get('label')
            if label in ['ET-Length', 'Edge-chip-depth', 'EE-Depth']:
                points = self.parse_points(polyline.get('points'))
                length = self._polyline_length(points)
                polylines[label] = length
        return polylines


def parse_all_annotations(
    annotations_dir: Path,
    images_dir: Optional[Path] = None,
    crop_to_upper_half: bool = True,
    manifest_path: Optional[Path] = None
) -> List[Annotation]:
    """
    Parse all XML files in annotations directory with progress indicators and validation.
    
    This function:
    1. Verifies annotation files exist and are parseable
    2. Tracks parsing statistics (success/failure)
    3. Shows progress indicators for large datasets
    4. Validates image-annotation correspondence
    
    Args:
        annotations_dir: Path to the directory containing XML annotation files
        images_dir: Path to the directory containing image files (optional if manifest_path is provided)
        crop_to_upper_half: Whether to crop defects to upper half of image
        manifest_path: Path to manifest.json file for S3 extraction pipeline (optional)
        
    Returns:
        List of parsed Annotation objects
        
    Raises:
        FileNotFoundError: If annotations directory doesn't exist or is empty
        ValueError: If images_dir is not provided when manifest_path is not provided
    """
    import logging
    from tqdm import tqdm
    
    # Get logger
    logger = logging.getLogger("armor_pipeline.parser")
    
    # Verify annotations directory exists
    if not annotations_dir.exists():
        error_msg = f"Annotations directory not found: {annotations_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Find all XML files
    xml_files = list(annotations_dir.glob("**/*.xml"))
    if not xml_files:
        error_msg = f"No XML annotation files found in {annotations_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Found {len(xml_files)} XML annotation files in {annotations_dir}")
    
    # Set up parser based on whether manifest_path is provided
    if manifest_path is not None:
        # Use manifest mappings to resolve image paths
        try:
            logger.info(f"Loading parser from manifest: {manifest_path}")
            parser = XMLAnnotationParser.load_from_manifest(manifest_path, crop_to_upper_half)
            # If images_dir is provided, override the one from manifest
            if images_dir is not None:
                parser.image_dir = images_dir
                logger.info(f"Overriding image directory from manifest with: {images_dir}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading manifest file {manifest_path}: {e}")
            raise
    else:
        # Use traditional directory scanning approach
        if images_dir is None:
            error_msg = "images_dir must be provided when manifest_path is not provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Using traditional directory scanning with image directory: {images_dir}")
        parser = XMLAnnotationParser(crop_to_upper_half)
    
    annotations = []
    parsing_stats = {
        "total": len(xml_files),
        "success": 0,
        "error": 0,
        "images_found": 0,
        "images_not_found": 0,
        "defects_parsed": 0
    }

    # Iterate through all XML files with progress bar
    for xml_file in tqdm(xml_files, desc="Parsing annotations", unit="file"):
        try:
            # When using manifest, images_dir is already set in the parser
            if manifest_path is not None:
                ann = parser.parse_file(xml_file)
            else:
                ann = parser.parse_file(xml_file, images_dir)
            
            # Track image-annotation correspondence
            if ann.image_path.exists():
                parsing_stats["images_found"] += 1
            else:
                parsing_stats["images_not_found"] += 1
                
            # Track defects parsed
            parsing_stats["defects_parsed"] += len(ann.defects)
            
            annotations.append(ann)
            parsing_stats["success"] += 1
        except Exception as e:
            parsing_stats["error"] += 1
            logger.error(f"Error parsing {xml_file}: {e}")
            # Don't print to console to avoid cluttering the progress bar
    
    # Log parsing statistics
    logger.info(f"Parsing completed: {parsing_stats['success']} successful, {parsing_stats['error']} failed")
    logger.info(f"Image correspondence: {parsing_stats['images_found']} found, {parsing_stats['images_not_found']} not found")
    logger.info(f"Total defects parsed: {parsing_stats['defects_parsed']}")
    
    if parsing_stats["error"] > 0:
        logger.warning(f"{parsing_stats['error']} annotation files could not be parsed")
        
    if parsing_stats["images_not_found"] > 0:
        logger.warning(f"{parsing_stats['images_not_found']} annotations have missing image files")
    
    return annotations


def parse_merged_annotations_with_s3(
    xml_path: Path,  # Your local merged_annotations_MINUS-TAM07-Q3-20250509.xml
    s3_loader: S3ImageLoader,
    crop_to_upper_half: bool = True
) -> List[Annotation]:
    """
    Parse a merged XML annotation file and match annotations with images stored in S3.
    
    This function:
    1. Parses the merged XML file
    2. Gets a list of S3 images from the S3 loader
    3. Creates a mapping of image names to S3 keys
    4. Creates annotations with S3 paths for matching images
    5. Parses defects for each annotation
    
    Args:
        xml_path: Path to the local merged XML annotation file
        s3_loader: Instance of S3ImageLoader to access S3 images
        crop_to_upper_half: Whether to crop defects to upper half of image (default: True)
        
    Returns:
        List of Annotation objects with S3 paths
        
    Example:
        ```python
        from armor_pipeline.data.parser import parse_merged_annotations_with_s3
        from armor_pipeline.data.s3_image_loader import S3ImageLoader
        from pathlib import Path
        
        # Initialize S3 loader
        s3_loader = S3ImageLoader()
        
        # Parse merged XML file
        xml_path = Path("merged_annotations.xml")
        annotations = parse_merged_annotations_with_s3(xml_path, s3_loader)
        
        # Use annotations with S3 paths
        for ann in annotations:
            print(f"Image: {ann.image_path}, Defects: {len(ann.defects)}")
        ```
    """
    
    parser = XMLAnnotationParser(crop_to_upper_half)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    s3_images = s3_loader.get_image_list()
    
    # Create mapping of image names to S3 keys
    image_map = {Path(s3_key).name: s3_key for s3_key in s3_images}
    
    # Parse each annotation
    for ann_elem in root.findall('.//annotation'):
        image_name = ann_elem.find('.//filename').text
        
        if image_name in image_map:
            s3_key = image_map[image_name]
            
            # Create annotation with S3 path
            annotation = Annotation(
                image_path=Path(s3_key),  # Store S3 key as path
                defects=[]
            )
            
            # Parse defects
            for obj in ann_elem.findall('.//object'):
                defect = parser._parse_object(obj)
                if defect and parser._is_in_upper_half(defect):
                    annotation.defects.append(defect)
            
            annotations.append(annotation)
    
    logger = logging.getLogger("armor_pipeline.parser")
    logger.info(f"Parsed {len(annotations)} annotations from merged XML")
    return annotations