"""SetupTools main script."""
import os
import versioneer
from setuptools import find_packages, setup

readme = os.path.join(os.path.dirname(__file__), 'README.md')

with open(os.path.join(os.path.dirname(__file__),
                       'requirements.txt')) as requirements:
    install_requires = requirements.readlines()

setup(name='TFM Ismael Miled',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="Study of different ML Deep Learning Tools for Finance ",
      long_description=open(readme).read(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: Other/Proprietary License',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Build Tools',
      ],
      keywords='Finance Vgg Prediction Classification Deep Learning ',
      author='Ismael Miled',
      author_email='ismael.miled@amundi.com',
      url='Some URL',
      license='All rights reserved',
      packages=find_packages(exclude=['tests']),
      install_requires=install_requires,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      include_package_data=True,
      zip_safe=False
      )
