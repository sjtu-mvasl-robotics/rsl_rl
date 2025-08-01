#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Copyright 2025 Shanghai Jiao Tong University, MVASL Lab, Yifei Yao
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="2.3.1.dev2",
    packages=find_packages(),
    author="Yifei Yao",
    maintainer="Yifei Yao",
    maintainer_email="godchaser@sjtu.edu.cn",
    url="https://github.com/sjtu-mvasl-robotics/rsl_rl",
    license="BSD-3-Clause",
    description="Enhanced RL algorithms for GBC project - based on RSL RL framework",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
        "onnx",
        "peft",
        "transformers",
    ],
)
