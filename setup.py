import os

import setuptools

def parse_requirements(filename):
    with open(filename, 'r') as f:
        # Read all lines and filter out empty lines or comments
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]


setuptools.setup(
    name="tensor",
    version="0.0.1",
    author="Zixing QIU",
    author_email="zixing.qiu.sci@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=parse_requirements('requirements.txt'),
)