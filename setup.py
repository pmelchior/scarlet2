from setuptools import setup

long_description = open('README.md').read()

setup(
    name='scarlet2',
    version='0.3.0',
    description='scarlet2: Astronomical source modeling in jax',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pmelchior/scarlet2',
    author='Peter Melchior',
    author_email='melchior@astro.princeton.edu',
    license='MIT',
    packages=['scarlet2'],
    install_requires=[
        'equinox',
        'jax',
        'astropy',
        'numpy',
        'matplotlib',
        'varname'
    ],
)
