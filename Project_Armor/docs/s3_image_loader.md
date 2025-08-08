# S3 Image Loader

This document describes the S3 image loader functionality for Project Armor, which enables loading images from Amazon S3 storage.

## Overview

The S3 image loader provides a way to load images directly from Amazon S3 storage, with local caching for improved performance. This is particularly useful when working with large datasets stored in S3, as it allows the pipeline to access images without downloading the entire dataset.

## Features

- **S3 Integration**: Load images directly from Amazon S3 buckets
- **Local Caching**: Cache images locally to improve performance for repeated access
- **LRU Caching**: In-memory caching for frequently accessed images
- **Prefetching**: Proactively download images to the cache
- **Cache Management**: Clear the cache to manage disk space

## Implementation

The S3 image loader is implemented in the `S3ImageLoader` class in `armor_pipeline/data/s3_image_loader.py`. It uses the boto3 library to interact with Amazon S3 and provides methods for loading images and managing the cache.

### S3ImageLoader Class

```python
class S3ImageLoader:
    """Simple S3 image loader with caching for Project Armor"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the S3 image loader.
        
        Args:
            cache_dir: Path to the cache directory. If None, defaults to './s3_cache'
        """
        # ...
```

#### Key Methods

- `load_image(s3_key: str) -> Optional[np.ndarray]`: Load an image from S3 with local caching
- `get_image_list(prefix: str = "ProductionLineImages/") -> List[str]`: List all images in the S3 bucket
- `clear_cache(older_than_days: Optional[int] = None)`: Clear the local cache
- `prefetch_images(s3_keys: List[str])`: Prefetch multiple images into the cache

## Integration with ContactLensDataset

The S3 image loader is integrated with the `ContactLensDataset` class to enable loading images from S3 during training and inference. The integration is implemented in the `_load_image()` method of the `ContactLensDataset` class.

### Usage with ContactLensDataset

To use the S3 image loader with the `ContactLensDataset` class, set the `use_s3` parameter to `True` when creating the dataset:

```python
dataset = ContactLensDataset(
    annotations=annotations,
    transform=transform,
    mode="train",
    use_s3=True  # Enable S3 loading
)
```

The dataset will then use the S3 image loader to load images with paths starting with "s3://".

## Configuration

The S3 image loader uses the following environment variables for configuration:

- `S3_ACCESS_KEY`: AWS access key ID
- `S3_SECRET_KEY`: AWS secret access key
- `DEFAULT_REGION`: AWS region (default: "us-east-2")
- `DEFAULT_BUCKET`: S3 bucket name (default: "ali-tam17")

These environment variables should be set before using the S3 image loader.

## Usage Examples

### Basic Usage

```python
from armor_pipeline.data.s3_image_loader import S3ImageLoader

# Set environment variables for S3 access
import os
os.environ['S3_ACCESS_KEY'] = 'your_access_key'
os.environ['S3_SECRET_KEY'] = 'your_secret_key'
os.environ['DEFAULT_REGION'] = 'us-east-2'
os.environ['DEFAULT_BUCKET'] = 'ali-tam17'

# Initialize loader
s3_loader = S3ImageLoader()

# List available images
image_keys = s3_loader.get_image_list(prefix="ProductionLineImages/")
print(f"Found {len(image_keys)} images")

# Load a specific image
if image_keys:
    image = s3_loader.load_image(image_keys[0])
    if image is not None:
        print(f"Loaded image with shape: {image.shape}")
```

### Usage with ContactLensDataset

```python
from armor_pipeline.data.dataset import ContactLensDataset

# Create dataset with S3 support
dataset = ContactLensDataset(
    annotations=annotations,
    transform=transform,
    mode="train",
    use_s3=True  # Enable S3 loading
)

# The dataset will now use S3 for images with paths starting with "s3://"
```

### Prefetching Images

```python
# List available images
image_keys = s3_loader.get_image_list(prefix="ProductionLineImages/")

# Prefetch the first 10 images
s3_loader.prefetch_images(image_keys[:10])

# Now loading these images will be faster as they're already in the cache
```

### Managing the Cache

```python
# Clear the entire cache
s3_loader.clear_cache()

# Clear only files older than 7 days
s3_loader.clear_cache(older_than_days=7)
```

## Testing

A test script is provided in `scripts/test_s3_image_loader.py` to verify the functionality of the S3 image loader. The script includes tests for both real S3 access and mock testing for environments without S3 access.

To run the tests:

```bash
python scripts/test_s3_image_loader.py
```

## Troubleshooting

### Missing AWS Credentials

If you encounter errors related to missing AWS credentials, make sure the environment variables are set correctly:

```bash
export S3_ACCESS_KEY=your_access_key
export S3_SECRET_KEY=your_secret_key
export DEFAULT_REGION=us-east-2
export DEFAULT_BUCKET=ali-tam17
```

### S3 Access Denied

If you encounter "Access Denied" errors, check that:
- The AWS credentials have the necessary permissions to access the S3 bucket
- The bucket name is correct
- The region is correct

### Image Loading Failures

If images fail to load:
- Check that the S3 key is correct
- Verify that the image exists in the S3 bucket
- Check that the image format is supported (BMP, JPG, PNG)
- Ensure the cache directory is writable

## Conclusion

The S3 image loader provides a convenient way to load images from Amazon S3 storage, with local caching for improved performance. It integrates seamlessly with the existing `ContactLensDataset` class, allowing the pipeline to work with datasets stored in S3 without downloading the entire dataset.