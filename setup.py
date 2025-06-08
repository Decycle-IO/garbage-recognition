from setuptools import setup, find_packages

setup(
    name='object_recognition',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines()
)
