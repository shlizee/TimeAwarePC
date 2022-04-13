from setuptools import setup
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='timeawarepc',
    url='https://github.com/shlizee/TimeAwarePC',
    author='Rahul Biswas (GitHub: biswasr)',
    author_email='rbiswas1@uw.edu',
    # Needed to actually package something
    packages=['timeawarepc'],
    # Needed for dependencies
    install_requires=['numpy','pickle','pandas','rpy2','networkx','scipy'],
    # *strongly* suggested for sharing
    version='alpha 1.0',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)