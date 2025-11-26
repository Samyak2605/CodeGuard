from setuptools import setup, find_packages

setup(
    name="codeguard",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "pandas",
        "numpy",
    ],
)
