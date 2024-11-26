from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="factorizer",
    version="0.0.1",
    author="Pooya Ashtari",
    author_email="pooya.ash@gmail.com",
    description="Factorizer - PyTorch",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pashtari/factorizer",
    project_urls={
        "Bug Tracker": "https://github.com/pashtari/factorizer/issues",
        "Source Code": "https://github.com/pashtari/factorizer",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "machine learning",
        "deep learning",
        "image segmentation",
        "medical image segmentation",
        "Factorizer",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml==6.0",
        "numpy==1.23.3",
        "sympy==1.11.1",
        "scipy==1.9.1",
        "pandas==1.4.4",
        "scikit-image==0.19.3",
        "torch==1.11",
        "torchvision==0.12.0",
        "pytorch-lightning==1.6.5",
        "torchmetrics==0.9.3",
        "einops==0.4.1",
        "opt_einsum==3.3.0",
        "networkx==2.8.6",
        "plotly==5.10.0",
        "itk",
        "nibabel==4.0.2",
        "monai==0.9.0",
        "performer-pytorch==1.1.4",
    ],
)
