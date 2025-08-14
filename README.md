# RSL RL for GBC

Re-implemented some components for better imitation learning. For details please refer to [GBC](https://github.com/sjtu-mvasl-robotics/GBC.git)

**!!! Important !!!**:

~~This version (`2.x.x-devx`) is not compatible with `legged_gym` since the interface (original interface) has been severely modified by ETHZ and NVIDIA. To install the version for `legged_gym`, please use the `1.0.2` branch~~. 

* Notice Update 2025-05-14:

For that NVIDIA updates `Isaac Lab` and `rsl_rl` lib so frequently, I am no longer capable of maintaining the compatibility with `legged_gym`.

The `1.0.2` branch is now deprecated.

## RSL RL MM Series Interface (2025-05-06)

The RSL RL MM Series Interface is a new interface for the RSL RL framework. It is built upon `envs.MMVecEnv` and called only by  `rsl_rl.runners.on_policy_runner_mm`. The changes to the interface are as follows:

* get_observations
    * former: `get_observations(self) -> tuple[torch.Tensor, dict]:`
    * new: `get_observations(self) -> tuple[torch.Tensor, dict]:`
    * Description: This function requires to return **direct observations** only. Former `rsl_rl` just require to set this function up for observations, but now we require users to also implement `get_reference_observations` even if no reference observations are used (just return `None`).
* get_reference_observations
    * former: `None`
    * new: `tuple[tuple[torch.Tensor, torch.Tensor] | None, dict]`
    * Description: This function requires to return **reference observations** only. The first element (tuple) contains `reference_observations` of shape (num_envs, *ref_obs_shape) (use zeros to fill up environments where no references are provided) and `reference_masks` of shape (num_envs,) indicating which environments have reference observations. The second element is a dictionary containing additional information. If no reference observations are used, return `None`, {}.
* reset
    * former: `reset(self) -> tuple[torch.Tensor, dict]:`
    * new: `reset(self) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, dict]`
    * Description: New reset also requires to reset reference observation buffers.
* step
    * former: `step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:`
    * new: `step(self, actions: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor, torch.Tensor, dict]:`
    * Description: New step also requires to return reference observations and masks.

For update in `rsl_rl.runners.on_policy_runner_mm`, please check the `rsl_rl.runners.on_policy_runner_mm` module. You can simply refer to our project [GBC](https://github.com/sjtu-mvasl-robotics/GBC) for recommended usage.

## Introduction
This is a unofficial fork of `rsl_rl` lib. Maintained by [@mvasl-robotics](https://github.com/sjtu-mvasl-robotics). However, this fork will no longer be maintained once I graduate from SJTU.

This fork is used in our imitation learning project [GBC](https://github.com/sjtu-mvasl-robotics/GBC).

## Original Introduction


Fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac GYM.

| :zap:        The `algorithms` branch supports additional algorithms (SAC, DDPG, DSAC, and more)! |
| ------------------------------------------------------------------------------------------------ |

Only PPO is implemented for now. More algorithms will be added later.
Contributions are welcome.

**Maintainer**: David Hoeller and Nikita Rudin <br/>
**Affiliation**: Robotic Systems Lab, ETH Zurich & NVIDIA <br/>
**Contact**: rudinn@ethz.ch

## Setup

Following are the instructions to setup the repository for your workspace:

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

The framework supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of the PPO, please check: [dummy_config.yaml](config/dummy_config.yaml) file.


## Contribution Guidelines

For documentation, we adopt the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. We use [Sphinx](https://www.sphinx-doc.org/en/master/) for generating the documentation. Please make sure that your code is well-documented and follows the guidelines.

We use the following tools for maintaining code quality:

- [pre-commit](https://pre-commit.com/): Runs a list of formatters and linters over the codebase.
- [black](https://black.readthedocs.io/en/stable/): The uncompromising code formatter.
- [flake8](https://flake8.pycqa.org/en/latest/): A wrapper around PyFlakes, pycodestyle, and McCabe complexity checker.

Please check [here](https://pre-commit.com/#install) for instructions to set these up. To run over the entire repository, please execute the following command in the terminal:


```bash
# for installation (only once)
pre-commit install
# for running
pre-commit run --all-files
```

### Useful Links

Environment repositories using the framework:

* `Legged-Gym` (built on top of NVIDIA Isaac Gym): https://leggedrobotics.github.io/legged_gym/
* `Orbit` (built on top of NVIDIA Isaac Sim): https://isaac-orbit.github.io/

## Citation

If you use this software in your research, please cite it as follows:

```bibtex
@software{rsl_rl_gbc_2025,
  title = {RSL RL for GBC: Enhanced Framework for Whole-Body Humanoid Imitation},
  author = {Yao, Yifei and Luo, Chengyuan},
  url = {https://github.com/sjtu-mvasl-robotics/rsl_rl},
  version = {2.3.1.dev2},
  year = {2025},
  month = {8},
  day = {1},
  note = {Enhanced version of RSL RL for GBC project. Original RSL RL by ETH Zurich and NVIDIA.}
}
```

For the original RSL RL framework, please cite:
```bibtex
@misc{rsl_rl_original,
  title = {RSL RL},
  author = {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  url = {https://github.com/leggedrobotics/rsl_rl},
  year = {2021}
}
```

You can also use the DOI from Zenodo (available after creating a release): [![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://doi.org/YOUR_DOI_HERE)

## License

This enhanced version is licensed under BSD-3-Clause:
- **This Version**: Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab, Yifei Yao & Chengyuan Luo

**Original Framework Attribution**:
- Copyright (c) 2021, ETH Zurich, Nikita Rudin
- Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES
- Original license: `licenses/original/LICENSE-ORIGINAL`
- Original repository: https://github.com/leggedrobotics/rsl_rl

This work contains substantial modifications specifically designed for the GBC 
(Generalized Behavior-Cloning Framework for Whole-Body Humanoid Imitation) project, 
while preserving compatibility with the original framework's interface where possible.
