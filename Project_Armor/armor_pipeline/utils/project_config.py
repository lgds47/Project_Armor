"""
Project Configuration module for Project Armor.

This module provides functionality to manage project paths and directories.
"""

import os
from pathlib import Path
from typing import Optional


class ProjectConfig:
    """Configuration class for managing project paths and directories."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the project configuration.
        
        Args:
            base_path: Base path for the project. If None, uses the environment variable
                      PROJECT_ARMOR_ROOT or defaults to the current directory.
        """
        self.base_path = base_path or Path(os.environ.get('PROJECT_ARMOR_ROOT', '.'))
        self.data_path = self.base_path / 'data'
        self.image_path = self.data_path / 'images'
        self.annotation_path = self.data_path / 'annotations'
        
        # Create directories if they don't exist
        for path in [self.data_path, self.image_path, self.annotation_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'ProjectConfig':
        """
        Create a ProjectConfig instance using environment variables.
        
        Environment variables:
            PROJECT_ARMOR_ROOT: Base path for the project
            DATA_ROOT: Path to the data directory (overrides base_path/data)
            IMAGE_PATH: Path to the image directory (overrides data_path/images)
            ANNOTATION_PATH: Path to the annotation directory (overrides data_path/annotations)
        
        Returns:
            ProjectConfig instance configured from environment variables
        """
        config = cls(base_path=Path(os.environ.get('PROJECT_ARMOR_ROOT', '.')))
        
        # Override paths if environment variables are set
        if 'DATA_ROOT' in os.environ:
            config.data_path = Path(os.environ['DATA_ROOT'])
        
        if 'IMAGE_PATH' in os.environ:
            config.image_path = Path(os.environ['IMAGE_PATH'])
        
        if 'ANNOTATION_PATH' in os.environ:
            config.annotation_path = Path(os.environ['ANNOTATION_PATH'])
        
        # Create directories if they don't exist
        for path in [config.data_path, config.image_path, config.annotation_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        return config
    
    def get_path(self, path_type: str) -> Path:
        """
        Get a path by type.
        
        Args:
            path_type: Type of path to get ('base', 'data', 'image', 'annotation')
        
        Returns:
            Path object for the requested path type
        
        Raises:
            ValueError: If path_type is not recognized
        """
        if path_type == 'base':
            return self.base_path
        elif path_type == 'data':
            return self.data_path
        elif path_type == 'image':
            return self.image_path
        elif path_type == 'annotation':
            return self.annotation_path
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return (
            f"ProjectConfig:\n"
            f"  base_path: {self.base_path}\n"
            f"  data_path: {self.data_path}\n"
            f"  image_path: {self.image_path}\n"
            f"  annotation_path: {self.annotation_path}"
        )


# Convenience function to get a ProjectConfig instance
def get_project_config(base_path: Optional[Path] = None) -> ProjectConfig:
    """
    Get a ProjectConfig instance.
    
    Args:
        base_path: Base path for the project. If None, uses environment variables
                  or defaults.
    
    Returns:
        ProjectConfig instance
    """
    if base_path:
        return ProjectConfig(base_path=base_path)
    else:
        return ProjectConfig.from_env()