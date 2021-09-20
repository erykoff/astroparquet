from setuptools import setup, find_packages


name = 'astroparquet'

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    description='Storing astropy tables (with units) in parquet',
    author='Eli Rykoff',
    author_email='erykoff@stanford.edu',
    url='https://github.com/erykoff/astroparquet',
    install_requires=['numpy', 'astropy', 'pyarrow'],
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
