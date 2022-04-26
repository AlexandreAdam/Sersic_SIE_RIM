from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
	name="carrim",
    author="Alexandre Adam",
    author_email="alexandre.adam@umontreal.ca",
    short_description="",
    long_description=long_description,
    long_description_content_type='text/markdown',
	version="0.1",
	description="",
	packages=find_packages(),
	python_requires=">=3.6"
)
