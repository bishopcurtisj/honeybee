from setuptools import find_packages, setup

setup(
    name="honeybee",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "jax",
    ],
)
