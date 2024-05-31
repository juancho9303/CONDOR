from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name='condor_utils',
    version="0.0.1",
    author='Juan Espejo',
    author_email='juancho9303@gmail.com',
    description='CONDOR utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
    ],
    package_dir={"condor_utils": "condor_utils"},
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
)
