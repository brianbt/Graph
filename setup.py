from setuptools import setup, find_packages


setup(
    name='graph',
    version='0.1',
    license='MIT',
    author="Chan Cheong",
    author_email='brianchan.xd@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/brianbt/Graph',
    keywords='python graph',
    install_requires=[
          'numpy',
      ],

)
