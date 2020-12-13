from setuptools import setup

setup(name='gammli',
      version='1.0',
      description='Explainable Recommendation Systems by Generalized Additive Models with Manifest and Latent Interactions',
      url='https://github.com/gyf9712/GAMMLI',
      author='Yifeng Guo',
      author_email='gyf9712@hku.hk',
      license='MIT',
      packages=['gammli'],
      install_requires=['matplotlib>=2.2.2','tensorflow>=2.0.0', 'numpy>=1.15.2'],
      zip_safe=False)