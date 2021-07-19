import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scnndeconv",
    version="0.0.1",
    author="Sylvain Prigent",
    author_email="sylvain.prigent@inria.fr",
    description="Deep learning models for image denoising and deconvolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sylvainprigent",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "pytorch>=1.0.1",
        "lightning>=1.0.1",
    ],
)
