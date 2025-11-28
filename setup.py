from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="tryondiffusion",
    version="0.1.0",
    description="Try-on diffusion model implementation",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)