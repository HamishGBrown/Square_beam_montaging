from setuptools import setup, find_packages

setup(
    name="SquareBeamMontaging",
    version="1.0.0",
    description="A tool for stitching square beam montage tomography data.",
    author="HamishGBrown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "tqdm",
        "scikit-image",
        "pypng",
        "Pillow",
        "matplotlib",
        "mrcfile",
        "argparse",
    ],
    entry_points={
        "console_scripts": [
            "stitch_square_beam = stitch:main",  # Assuming the main entry point is in the stitch.py file
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
