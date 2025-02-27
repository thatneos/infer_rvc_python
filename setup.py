import os
import platform
import pkg_resources
import setuptools
from setuptools import find_packages, setup
from pkg_resources import parse_version

# Read the long description with explicit encoding
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup_kwargs = {
    "name": "infer_rvc_python",
    "version": "1.2.0",
    "description": "Python wrapper for fast inference with rvc",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "python_requires": ">=3.10",
    "author": "R3gm",
    "url": "https://github.com/R3gm/infer_rvc_python",
    "license": "MIT",
    "packages": find_packages(),
    "package_data": {"": ["*.txt", "*.rep", "*.pickle"]},
    "install_requires": [
        "torch",
        "torchaudio",
        "gradio",
        "yt-dlp",
        "audio-separator[gpu]==0.28.5",
        "praat-parselmouth>=0.4.3",
        "pyworld==0.3.2",
        "faiss-cpu==1.7.3",
        "torchcrepe==0.0.23",
        "ffmpeg-python>=0.2.0",
        "fairseq==0.12.2",
        "transformers",
        "typeguard==4.2.0",
        "soundfile",
        "numpy",
    ],
    "include_package_data": True,
    "extras_require": {
        "all": [
            "scipy",
            "numba==0.56.4",
            "edge-tts"
        ]
    },
}

# Conditionally include the "readme" parameter if supported
if parse_version(setuptools.__version__) >= parse_version("68.0.0"):
    setup_kwargs["readme"] = "README.md"

setup(**setup_kwargs)
