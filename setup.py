from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="multimodal-supernovae",
    version="0.1",
    description="",
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Machine learning :: Supernovea :: Multimodal",
    ],
    keywords="Machine learning, Supernovea, Multimodal",
    author="Thomas Helfer, Gemma Zhang",
    author_email="thomashelfer@live.de",
    license="MIT",  
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
