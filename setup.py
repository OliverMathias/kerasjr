from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
  name = 'kerasjr',         # How you named your package folder (MyLib)
  packages = ['kerasjr'],   # Chose the same as "name"
  version = '0.1.5',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A python package to create and train simple dense neural networks.',   # Give a short description about your library
  author = 'Oliver Mathias',                   # Type in your name
  author_email = 'mathias@chapman.edu',      # Type in your E-Mail
  url = 'https://github.com/OliverMathias/kerasjr',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/OliverMathias/kerasjr/archive/v0.1.5.tar.gz',    # I explain this later on
  keywords = ['Dense', 'model', 'machine learning', 'Deep Learning', 'prediction'],   # Keywords that define your package best
  install_requires=['numpy'],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.7',
  ],

  long_description=long_description,
   long_description_content_type="text/markdown"
)
