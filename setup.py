from setuptools import setup, find_packages

# with open('requirements.txt') as f:
#     required = f.read().splitlines()
# print(required)
from csdid._version import __version
print(__version)

setup(
  name = 'csdidspark',
  version=__version,
  url='https://github.com/d2cml-ai/csdid-pyspark',
  author='D2CML Team, Alexander Quispe, Carlos Guevara, Jhon Flores',
  keywords=['Causal inference', 'Research'],
  license="MIT",
  description='Difference in Difference in Python with PySpark',
  classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
    ],
  install_requires=[
	'pyspark',
	'joblib',
	'findspark',
	'matplotlib',
	'numpy',
	'pandas',
	'tqdm',
	'scipy'
  ],
  packages=find_packages(),
#   package_data={
    # 'data': ['data/*'],
    # 'configs': ['configs/*']
#   }
)