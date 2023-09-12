#!/usr/bin/env python3
# Inspired from https://github.com/kennethreitz/setup.py
from pathlib import Path

from setuptools import setup, find_packages


NAME = 'arabic_vocalizer'
DESCRIPTION = 'Arabic vocalization models'
URL = 'https://github.com/nipponjo/'
EMAIL = ''
AUTHOR = ''
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '0.0.1'

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(

    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,

    packages=find_packages(),
    install_requires=['numpy', 'onnxruntime-gpu'],
    include_package_data=True,
    data_files=[('data', ['arabic_vocalizer/data/shakkala.onnx', 
                          'arabic_vocalizer/data/shakkelha.onnx']),
                ('license', ['arabic_vocalizer/LICENSE', 
                             'arabic_vocalizer/models/shakkala/LICENSE.md',
                             'arabic_vocalizer/models/shakkelha/LICENSE'])
               ], 

    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Scientific/Engineering :: Artificial Intelligence',   
        'Programming Language :: Python',        
        'License :: OSI Approved :: MIT License',
    ])