from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules = [Extension(
    'dataset_tools.utils._mask',
    sources=['dataset_tools/utils/maskApi.c', 'dataset_tools/utils/_mask.pyx'],
    include_dirs = [np.get_include()],
    extra_compile_args=[]
)]

setup(
    name='dataset-tools',
    version='0.1b',
    author='Wang Ming Rui',
    author_email='mingruimingrui@hotmail.com',
    packages=[
        'dataset_tools',
        'dataset_tools.datasets',
        'dataset_tools.transforms',
        'dataset_tools.samplers',
        'dataset_tools.utils'
    ],
    ext_modules=cythonize(ext_modules)
)
