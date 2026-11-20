from setuptools import find_packages, setup

setup(
    name='rdet',
    version='0.0.1',
    description='Modern object detectors as an mmdetection plugin, built for learning.',
    packages=find_packages(include=['rdet', 'rdet.*', 'mini_mm', 'mini_mm.*']),
    python_requires='>=3.8',
)
