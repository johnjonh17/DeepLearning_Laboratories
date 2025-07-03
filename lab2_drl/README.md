# Deep Reinforcement Learning Laboratory

This notebook is a comprehensive laboratory for experimenting with advanced Deep Reinforcement Learning (DRL) algorithms. The main focus is on implementing, improving, and analyzing REINFORCE and Deep Q-Learning (DQN) algorithms on classic control environments such as CartPole and LunarLander from OpenAI Gymnasium.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Lab Sections](#lab-sections)
  - [1. Environment Exploration](#1-environment-exploration)
  - [2. REINFORCE Algorithm](#2-reinforce-algorithm)
  - [3. Baselines and Value Networks](#3-baselines-and-value-networks)
  - [4. LunarLander with REINFORCE](#4-lunarlander-with-reinforce)
  - [5. Deep Q-Learning (DQN)](#5-deep-q-learning-dqn)
  - [6. (Optional) CarRacing Challenge](#6-optional-carracing-challenge)
- [Dependencies](#dependencies)
- [Usage Tips](#usage-tips)
- [References](#references)

---

## Overview

- **Goal:** Learn, implement, and analyze DRL algorithms, focusing on policy gradient methods (REINFORCE) and value-based methods (DQN).
- **Environments:** CartPole-v1, LunarLander-v3.
- **Skills:** Environment exploration, policy/value network design, reward analysis, baseline techniques, and advanced DRL evaluation.

---


## Getting Started

1. **Open the notebook** `DLA_Lab2_DRL.ipynb` in Jupyter or Google Colab.
2. **Install dependencies** as prompted in the notebook (e.g., Gymnasium, Box2D, SWIG).
3. **(If required)** Upload and extract the provided `cartpole_rl_package_updated.zip` package for modular code (policy, agent, utils).
4. **Run cells sequentially** and follow the instructions in each section.

---

## Lab Sections

### 1. Environment Exploration

- Inspect the CartPole environment: observation space, action space, and reward structure.
- Run a random agent to understand episode dynamics.

### 2. REINFORCE Algorithm

- **Original REINFORCE:** Train a policy network using the vanilla REINFORCE algorithm.
- **Improved Evaluation:** Modify training to periodically evaluate the agent over multiple episodes, collecting average total reward and episode length for robust analysis.

### 3. Baselines and Value Networks

- **No Baseline:** Train REINFORCE without any normalization.
- **Standardization Baseline:** Use mean and standard deviation of returns as a baseline (already implemented).
- **Learned Value Baseline:** Implement and train a separate value network (`ValueNet`) as a baseline for the policy gradient update.
- **Comparison:** Plot and compare learning curves for different baseline strategies.

### 4. LunarLander with REINFORCE

- Adapt the REINFORCE algorithm (with value baseline) to the more challenging LunarLander-v3 environment.
- Analyze and visualize agent performance.

### 5. Deep Q-Learning (DQN)

- **Q-Network:** Implement a deep neural network for Q-value estimation.
- **Replay Buffer:** Store and sample experience tuples for stable training.
- **Target Network:** Use a slowly-updated target network to stabilize learning.
- **Training:** Apply DQN to CartPole and LunarLander, plot learning curves.


## Dependencies

- Python 3.7+
- torch
- numpy
- matplotlib
- gymnasium (`pip install gymnasium`)
- Box2D (`pip install "gymnasium[box2d]"`)
- swig (for LunarLander on Colab)
- google.colab (for file upload in Colab)

---

## Usage Tips

- **Reproducibility:** Set random seeds for torch and numpy.
- **Evaluation:** Use periodic evaluation on separate environments for meaningful metrics.
- **Visualization:** Render agent behavior using RGB arrays and matplotlib animations.
- **Experimentation:** Try different hyperparameters, network architectures, and baseline strategies.

---

## References

- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- [Deep Q-Learning Paper](https://www.nature.com/articles/nature14236)


---

