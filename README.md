# Regret Bounds for Satisficing in Multi-Armed Bandit Problems

Code for the paper "Regret Bounds for Satisficing in Multi-Armed Bandit Problems" by Thomas Michel, Hossein Hajiabolhassan, and Ronald Ortner. This repository provides implementations of various algorithms for the satisficing multi-armed bandit problem and code to reproduce the experiments from the paper.

## Overview

In the standard multi-armed bandit problem, the goal is to maximize cumulative reward. In the satisficing setting, the learner is instead content with finding an arm whose expected reward is above a given satisfaction level S. This repository provides implementations of the algorithms presented in the paper and allows comparing them with baseline methods.

## Installation

The project requires the following dependencies:
- Python 3.6+
- NumPy
- Matplotlib
- SciPy
- seaborn (optional, for improved visualizations)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Code Structure

- **bandit.py**: Core implementations of bandit classes and algorithms:
  - Bandit classes: `Bandit`, `GaussianBandit`, `BernoulliBandit`
  - Main algorithm implementations: 
    - `algo1` (Simple-Sat, Algorithm 1)
    - `algo3` (Sat-UCB, Algorithm 2)
    - `algo3_old` (Sat-UCB+, Algorithm 3)
    - Variants of Sat-UCB: `algo3xucb`, `algo3xavg`
  - Baseline algorithms: `ucb`, `bernoulli_thompson`, `epsilon_greedy`
  - Comparison algorithms: `deterministic_ucl`, `satisfaction_mean_reward`, `ucb_alpha_elimination`
  
- **experiments.py**: Script to run experiments and visualize results
- **setting.py**: Simple module for global settings

## Reproducing Experiments

### Main Experiments from the Paper

To reproduce the main experiments from the paper, run:

```bash
python experiments.py
```

By default, the script will execute the experiment for comparing satisfaction levels in the realizable case. You can modify the experiment parameters at the beginning of the `experiments.py` file:

```python
# Toggle which algorithms to include
plot_satucb = True           # Enable Sat-UCB algorithm
plot_satucbplus = True       # Enable Sat-UCB+ algorithm
plot_satucb_variants = True  # Enable Sat-UCB variants
plot_other_algorithms = True # Enable comparison algorithms
plot_ucb_algorithm = True    # Enable UCB1 algorithm

# Toggle special experiments
experiment_multiple_satisfaction_levels = False
experiment_1_run = False
experiment_round_robin = False

# Choose experimental setting
reward_distribution = "gaussian"  # "gaussian" or "bernoulli"
realizable_case = True            # True for realizable case, False for non-realizable
setting_id = 1                    # 1, 2 or 3 (different arm distributions)
```

### Experiment Settings

The paper presents experiments in three different settings:

1. **Setting 1**: Arms with evenly spaced means from 0 to 1
2. **Setting 2**: Arms with means distributed as sqrt(i/n)
3. **Setting 3**: One arm with mean 1, all others with mean 0

For each setting, both realizable (S < μ*) and non-realizable (S > μ*) cases are considered.

### Algorithm Descriptions

- **Simple-Sat (Algorithm 1)**: A simple algorithm that exploits if the empirical best arm exceeds the satisfaction level, and explores uniformly at random otherwise.
- **Sat-UCB (Algorithm 2)**: An algorithm that uses confidence bounds to make decisions, achieving constant regret in the realizable case and logarithmic regret in the non-realizable case.
- **Sat-UCB+**: A simplified version of Sat-UCB without the random exploration phase.
- **Sat-UCB variants**: Variants of Sat-UCB that use different indices for arm selection.

## Reproducibility

The experiments use a fixed random seed to ensure reproducibility. You can change the seed by modifying the `seed` variable at the beginning of `bandit.py`:

```python
seed = 1  # Set to None for random seed
```

## Paper Reference

T. Michel, H. Hajiabolhassan, and R. Ortner, "Regret Bounds for Satisficing in Multi-Armed Bandit Problems," Transactions on Machine Learning Research, 2023.