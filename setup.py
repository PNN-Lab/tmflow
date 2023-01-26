from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="tmflow",
    version="0.4.0",
    description="Taylor map flow is a package for a 'flowly' construction and learning "
                "of polynomial neural networks (PNN) for time-evolving process prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PNN-Lab/tmflow",
    author="PNN Lab",
    author_email="golovkina.a@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="PNN, Taylor, ODE, TensorFlow",
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=[
        "tensorflow>=2.8,<2.10",
        "scipy~=1.6.2",
        "numpy~=1.20.3",
        "sympy~=1.8",
        "numba~=0.55.1",
    ],
    #   $ pip install tmflow[examples]
    extras_require={
        "examples": [
            "matplotlib~=3.3.4",
            "pandas~=1.3.0",
            "scikit-learn~=0.24.1",
        ],
        # "test": [""],
    },
    project_urls={
        "Bug Reports": "https://github.com/PNN-Lab/tmflow/issues",
        "Source": "https://github.com/PNN-Lab/tmflow",
        "Examples": "https://github.com/PNN-Lab/tmflow/tree/main/examples",
        "Changelog": "https://github.com/PNN-Lab/tmflow/blob/main/CHANGELOG.md"
    },
)
