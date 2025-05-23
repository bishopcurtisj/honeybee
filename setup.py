from setuptools import find_packages, setup

setup(
    name="honeybee",
    version="0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
