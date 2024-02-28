from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="GeneralRelativity",
    version="0.1",
    description="A translation of crucial parts of GRTL in torch for accelerated learning",
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Machine learning :: Physics :: Simulation :: General Relativity",
    ],
    keywords="Machine learning, Physics, Simulation, General Relativity",
    author="ThomasHelfer",
    author_email="thomashelfer@live.de",
    license="MIT",  # Updated to MIT License
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
