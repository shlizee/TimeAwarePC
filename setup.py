import setuptools

with open('README.md','r') as fh:
    README = fh.read()

VERSION = "1.2.1"

setuptools.setup(
    name = 'timeawarepc',
    version = VERSION,
    author = 'Rahul Biswas',
    author_email = 'rahul.biswas@ucsf.edu',  # optional but good
    description = 'Time-Aware PC Python Package',
    long_description= README,
    long_description_content_type = 'text/markdown',
    license="MIT",
    url='https://github.com/biswasr/TimeAwarePC',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'rpy2==3.5.11',
        'networkx',
        'scipy'
    ],
    python_requires='==3.10.*',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
    ],
)
