import setuptools


packages = [
    'pynccl',
    'pynccl.utils',
]

setuptools.setup(
    name='pynccl',
    version='0.1.2',
    author="Lance Lee",
    author_email="lancelee82@163.com",
    description="pynccl - python bindings for NVIDIA NCCL libraries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/lancelee82/pynccl",
    packages=packages,
    install_requires=[
        'numpy', 'numba'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
 )
