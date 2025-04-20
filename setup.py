from setuptools import setup, find_packages

setup(
    name="dataanalysistool",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.19.0",
        "scikit-learn>=1.4.0",
        "yfinance>=0.2.55",
        "flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "dataanalysis=dataanalysistool.ui.cli:main",
        ],
    },
    author="Data Analysis Tool Team",
    author_email="achs@agikara.site",
    description="A comprehensive Python-based data analysis tool with financial analysis capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DataAnalysisTool",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
)
