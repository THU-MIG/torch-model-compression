from setuptools import setup, find_packages

setup(
    name="torchpruner",
    version="0.1.0",
    description="torchvision",
    author="ZizhouJia",
    url="",
    packages=find_packages(where=".", exclude=(), include=("*",)),
    install_requires=[
        "torch>=1.7",
        "onnx>=1.6",
        "onnxruntime>=1.5",
        "scikit-learn",
        "tensorboardX>=1.8",
    ],
)
