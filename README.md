# RSL RL

Re-implemented some components for better imitation learning.

**!!! Important !!!**:
The version (`2.0.x-devx`) is not compatible with `legged_gym` since the interface (original interface) has been severely modified by ETHZ. To install the version for `legged_gym`, please use the `1.0.2` branch. 

If you see this markdown, you are currently on `1.0.2` branch.

## RSL RL MM Series Interface (2024-12-24)

The RSL RL MM Series Interface is a new interface for the RSL RL framework. It is built upon `envs.MMVecEnv` and called only by  `rsl_rl.runners.on_policy_runner_mm`. The changes to the interface are as follows:

I'm a little busy now, you can just refer to our implementation or check the files in the `rsl_rl` directory for interface changes.

For update in `rsl_rl.runners.on_policy_runner_mm`, please check the `rsl_rl.runners.on_policy_runner_mm` module. You can simply refer to our project [GBC](https://github.com/sjtu-mvasl-robotics/GBC) for recommended usage.

## Introduction


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
