from setuptools import setup, find_packages

setup(
    name="defer",
    version="0.0.1",
    packages=find_packages(
        exclude=["notebooks", "scripts", "tests", "density_functions", "README_files"]
    ),
)
