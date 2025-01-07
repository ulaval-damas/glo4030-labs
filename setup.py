from distutils.core import setup

setup(
    name='deeplib',
    version='0.1',
    packages=['deeplib'],
    install_requires=['torch', 'torchvision', 'pandas', 'ipykernel', 'matplotlib', 'scikit-learn', 'graphviz',
                      'ipython', 'gensim', 'numpy', 'scipy', 'Pillow', 'poutyne', 'einops', 'transformers[torch]'],
)
