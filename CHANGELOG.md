# Changelog - RSL RL for GBC

All notable changes to this enhanced version of RSL RL for the GBC project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.1.dev2] - 2025-08-01

### Added
- RSL RL MM Series Interface for better imitation learning support
- New `get_reference_observations` method for reference-based learning
- Enhanced environment interface with reference observation support
- Integration with GBC (Gradient-Based Control) project
- Student-teacher transformer architecture support
- Multi-modal PPO implementation
- Memory modules for neural networks
- Distillation algorithms for knowledge transfer

### Changed
- Updated interface to support both direct and reference observations
- Modified reset and step methods to handle reference observations
- Enhanced rollout storage for multi-modal data
- Improved actor-critic architectures with transformer support

### Fixed
- Compatibility issues with newer PyTorch versions
- Interface consistency across different environment types

### Deprecated
- Compatibility with `legged_gym` (1.0.2 branch deprecated as of 2025-05-14)

### Notes
- This version is maintained by SJTU MVASL Lab
- Built upon the original RSL RL framework from ETH Zurich and NVIDIA
- Designed for GPU-accelerated reinforcement learning
