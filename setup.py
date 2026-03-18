from setuptools import setup, find_packages

setup(
    name="siglino",
    version="0.1.0",
    description="SigLino: Agglomeration Mixture of Experts Vision Foundation Model",
    author="Sofian Chaybouti",
    packages=find_packages(include=["siglino", "siglino.*"]),
    python_requires=">=3.10",
    homepage="https://github.com/sofianchaybouti/siglino",
)