import os
from setuptools import setup
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gem'))
from version import __version__

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='gem',
    py_modules=['gem'],
    version=__version__,
    install_requires=[
        'degmo',
        'torch>=1.0',
        'torchvision',
        'numpy==1.16.0',
        'scipy',
        'moviepy',
        'matplotlib',
        'tb_nightly',
        'tqdm',
        'pandas',
        'gym',
        'gym[atari]',
        'dm_control',
    ],
    description="General Intelligent Machine based on World Model",
    author="Xingyuan Zhang",
    author_email="wizardicarus@gmail.com",
    url="https://github.com/IcarusWizard/GEM"
)