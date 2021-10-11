from setuptools import setup, find_packages
from codecs import open
import os
import re

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setup(
    name='condor_utils',
    version="0.0.1",
    author='Juan Espejo',
    author_email='juancho9303@gmail.com',
    description='CONDOR utillities',
    long_description=long_description,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        ],
    package_dir={"condor_utils": "condor_utils"},
    packages=find_packages(),
    include_package_data=True,
)