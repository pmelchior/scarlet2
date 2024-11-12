from setuptools import setup

setup(
    name='scarlet2',
    version='0.2.0',    
    description='Scarlet, all new and shiny',
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
