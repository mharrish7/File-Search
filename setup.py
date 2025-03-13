from setuptools import setup, find_packages

setup(
    name='file_search',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'qdrant-client',
        'sentence-transformers',
    ],
    entry_points={
        'console_scripts': [
            'fsearch = file_search:main',  
        ],
    },
)