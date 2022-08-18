# Regret Bounds for Satisficing in Multi-Armed Bandit Problems

Code for the experiments of the paper "Regret Bounds for Satisficing in Multi-Armed Bandit Problems". It allows to run different algorithms on two classes of multi-armed bandits : Bernoulli bandits and Gaussian bandits.

## Usage
The project use `numpy` and `matplotlib` as external library. They can be installed using
```pip install matplotlib numpy```

The file `bandit.py` contains different class of bandits as well as the code to run the algorithms considered in this study. Please refer to the comments at the beginning of the file for details.

The files `bernoulli_experiment.py` and `gaussian_experiment.py` reproduce the figures used in the paper. Each file runs all the experiments before showing the plots. It may take a few minutes. 

These files can be run using ```python3 bernoulli_experiment.py``` or ```python3 gaussian_experiment.py```

## Reproducibility
The seed used for the experiment can be set at the beginning of `bandit.py`. The random number generator is reset with this seed before each new experiment so two experiments with the same algorithm and the same parameters produce the same result. Choosing the value `1` as the seed allows to produce results similar to the ones presented in the paper.