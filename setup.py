from setuptools import setup

setup(
    name='phasehull',
    version='0.1.0',    
    description='Computing phase diagrams with a convex hull',
    url='https://github.com/dullemond/phasehull',
    author='Cornelis Dullemond',
    author_email='dullemond@uni-heidelberg.de',
    license='MIT',
    packages=['phasehull','mineral_systems'],
    install_requires=['scipy',
                      'numpy',
                      'matplotlib'
                      ],
)
