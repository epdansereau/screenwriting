from setuptools import setup, find_packages

setup(
    name='screenwriting',
    version='0.1.0',
    description='A screenplay parser and formatter supporting FDX and plaintext.',
    author='Your Name',
    author_email='you@example.com',
    packages=find_packages(),
    install_requires=[
        'pydantic>=1.10',
    ],
    python_requires='>=3.7',
)