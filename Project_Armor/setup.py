from setuptools import setup, find_packages

setup(
    name="armor_pipeline",
    version="1.0.0",
    description="J&J Contact Lens Defect Detection Pipeline",
    author="Senior SWE/Data Scientist",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
        "albumentations>=1.3.0",
        "ultralytics>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "armor-pipeline=cli:main",
        ],
    },
)