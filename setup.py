from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="molformer_regresssion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Read dependencies from requirements.txt
        req.strip() for req in open("requirements.txt").readlines() if req.strip()
    ],
    entry_points={
        "console_scripts": [
            "molreg=molformer_regression.main:main", 
        ],
    },
    python_requires=">=3.9",
)
