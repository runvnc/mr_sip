#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="mr_sip",
    version="1.0.0",
    description="MindRoot SIP phone integration plugin with voice transcription and TTS",
    author="MindRoot Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "baresipy>=1.0.0",
        "faster-whisper>=0.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
