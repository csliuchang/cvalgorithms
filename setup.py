#!/usr/bin/env python
import os
import subprocess
import time
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

import torch.nn.modules


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='deepcv',
        description='DeepSight Detection Toolbox and Benchmark',
        author='deepsightAI',
        author_email='@deepsight.com',
        keywords='computer vision, classsification egmentation and detection',
        url='https://git.deepsight.ai/DeepLearningGroup/deepcv',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'deepcv.opts': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        ext_modules=[
            make_cuda_ext(
                name='rnms_ext',
                module='opts.rnms',
                sources=['src/rnms_ext.cpp', 'src/rcpu/rnms_cpu.cpp'],
                sources_cuda=[
                    'src/rcuda/rnms_cuda.cpp', 'src/rcuda/rnms_kernel.cu'
                ]),
            make_cuda_ext(
                name='rbbox_geo_cuda',
                module='opts.rbbox_geo',
                sources=[],
                sources_cuda=[
                    'src/rbbox_geo_cuda.cpp', 'src/rbbox_geo_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)