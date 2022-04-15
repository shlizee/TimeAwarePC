import setuptools

with open('README.md','r') as fh:
    README = fh.read()

VERSION = "1.0.0"

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name = 'timeawarepc',
    version = VERSION,
    author = '',
    description = 'Time-Aware PC Python Package',
    long_description= README,
    long_description_content_type = 'text/markdown',
    install_requires=['numpy','pandas','rpy2','networkx','scipy'],
    url='https://github.com/shlizee/TimeAwarePC',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
