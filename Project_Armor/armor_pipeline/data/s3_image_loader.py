"""
S3 Image Loader module for Project Armor.

This module provides functionality to load images from S3 with local caching.
"""

import boto3
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict, List
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class S3ImageLoader:
    """Simple S3 image loader with caching for Project Armor"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the S3 image loader.
        
        Args:
            cache_dir: Path to the cache directory. If None, defaults to './s3_cache'
        """
        # SECURITY: Use environment variables for credentials
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            region_name=os.getenv('DEFAULT_REGION', 'us-east-2')
        )
        self.bucket = os.getenv('DEFAULT_BUCKET', 'ali-tam17')
        
        # Set up cache directory
        self.cache_dir = cache_dir if cache_dir else Path('./s3_cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized S3ImageLoader with cache directory: {self.cache_dir}")
        
    @lru_cache(maxsize=1000)
    def load_image(self, s3_key: str) -> Optional[np.ndarray]:
        """
        Load image from S3 with local caching.
        
        Args:
            s3_key: S3 object key for the image
            
        Returns:
            NumPy array containing the image, or None if loading failed
        """
        # Create cache path using the filename from the S3 key
        cache_path = self.cache_dir / Path(s3_key).name
        
        # Check cache first
        if cache_path.exists():
            logger.debug(f"Loading image from cache: {cache_path}")
            try:
                image = cv2.imread(str(cache_path))
                if image is None:
                    logger.warning(f"Cached image is corrupted: {cache_path}. Will reload from S3.")
                else:
                    return image
            except Exception as e:
                logger.warning(f"Error loading cached image {cache_path}: {e}. Will reload from S3.")
        
        # If not in cache or cache is corrupted, download from S3
        try:
            logger.debug(f"Downloading image from S3: {s3_key}")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            image_bytes = response['Body'].read()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(image_bytes)
            
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Failed to decode image from S3: {s3_key}")
                return None
                
            logger.debug(f"Successfully loaded and cached image: {s3_key}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load {s3_key} from S3: {e}")
            return None
    
    def get_image_list(self, prefix: str = "ProductionLineImages/") -> List[str]:
        """
        List all images in S3 bucket with the given prefix.
        
        Args:
            prefix: S3 prefix to filter objects
            
        Returns:
            List of S3 keys for images
        """
        logger.info(f"Listing images in S3 bucket {self.bucket} with prefix {prefix}")
        images = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                        images.append(obj['Key'])
            
            logger.info(f"Found {len(images)} images in S3 bucket")
            return images
        except Exception as e:
            logger.error(f"Error listing images in S3 bucket: {e}")
            return []
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear the local cache, optionally only removing files older than specified days.
        
        Args:
            older_than_days: If provided, only clear files older than this many days
        """
        import time
        from datetime import datetime, timedelta
        
        if older_than_days is not None:
            cutoff_time = (datetime.now() - timedelta(days=older_than_days)).timestamp()
            logger.info(f"Clearing cache files older than {older_than_days} days")
        else:
            cutoff_time = None
            logger.info("Clearing entire cache")
        
        count = 0
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                if cutoff_time is None or cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    count += 1
        
        logger.info(f"Cleared {count} files from cache")
    
    def prefetch_images(self, s3_keys: List[str]):
        """
        Prefetch multiple images into the cache.
        
        Args:
            s3_keys: List of S3 keys to prefetch
        """
        logger.info(f"Prefetching {len(s3_keys)} images")
        for s3_key in s3_keys:
            cache_path = self.cache_dir / Path(s3_key).name
            if not cache_path.exists():
                try:
                    # Download but don't decode
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                    image_bytes = response['Body'].read()
                    
                    with open(cache_path, 'wb') as f:
                        f.write(image_bytes)
                        
                    logger.debug(f"Prefetched: {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to prefetch {s3_key}: {e}")