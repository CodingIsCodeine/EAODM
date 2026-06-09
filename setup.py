from setuptools import find_packages, setup

setup(
    name="m2mvt_daadx",
    version="1.0",
    author="M2MVT Reproduction",
    packages=find_packages(exclude=("configs", "tools")),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "fvcore",
        "av",           # PyAV for video decoding
        "opencv-python",
        "scikit-learn",
        "numpy",
        "pyyaml",
        "tqdm",
    ],
)
