from setuptools import setup, find_packages

setup(
    name="honeybee",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",  
        "jax",     
    ],
)