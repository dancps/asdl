from setuptools import setup, find_packages

setup(
    name='asdl',
    version='0.1',
    license='MIT License',
    long_description=open('README.md').read(),
    url='https://github.com/dancps/asdl',
    author='Danilo Calhes',
    packages=find_packages(),
    #entry_points={'console_scripts': ['asdl=asdl.asdl:main',],},
)