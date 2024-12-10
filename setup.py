from setuptools import setup, find_packages

setup(
    name="fft_uber",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
    ],
    python_requires=">=3.7",
)
