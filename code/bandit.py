# Regret Bounds for Satisficing in Multi-Armed Bandit Problems
#
# This file contains the implementation of bandit classes and algorithms 
# from the paper "Regret Bounds for Satisficing in Multi-Armed Bandit Problems"
# 
# The bandits take parameters to initialize their reward distributions,
# and this behavior can be overwritten by specifying the desired distributions directly.
#
# The algorithms are designed to run a simulation once from scratch for a given number of steps.
# Each algorithm function takes as input a bandit instance, a satisfaction level, and a number of steps.
# An algorithm-specific parameter can also be passed to tune the algorithm.
# The functions return different metrics: rewards, regrets, and expected rewards.
#
# The experiment function can be used to run algorithms multiple times and/or on different bandit instances
# and returns the average of the metrics.
#
# Set the seed variable below to an integer value to reproduce results from the paper (seed=1)
# This seed is used to reset the random number generator before each new experiment

import setting
import numpy as np
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt

# Global seed for reproducibility
seed = 1  # Set to None for random seed

# Simple CLI progressbar that doesn't require a particular library 
def progressbar(it, prefix="", size=60, file=sys.stdout):
    """Display a progress bar for the given iterable"""
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()

#-----------------------------------------------------------------------
# Bandit Classes
#-----------------------------------------------------------------------

class Bandit:
    """
    Parent class for bandits.
    Arms always return the same values.
    If means are unspecified, they are chosen uniformly in the given range.
    """
    def __init__(self, nb_arm, range=[0, 1], means=None):
        self.nb_arm = nb_arm
        if means is None:
            self.means = np.random.rand(nb_arm) * (range[1] - range[0]) + range[0]
        else:
            self.means = means

    def pull(self, arm):
        return self.means[arm]


class GaussianBandit(Bandit):
    """
    Bandit with Gaussian distributed rewards.
    If unspecified, the standard deviations are set to 1 by default.
    """
    def __init__(self, nb_arm, range=[0, 1], means=None, sigmas=None):
        super().__init__(nb_arm, range=range, means=means)
        if sigmas is not None:
            self.sigmas = sigmas
        else:
            self.sigmas = np.ones(nb_arm)

    def pull(self, arm):
        return np.random.randn() * self.sigmas[arm] + self.means[arm]


class BernoulliBandit(Bandit):
    """Bandit with Bernoulli distributed rewards."""
    def __init__(self, nb_arm, range=[0, 1], threshold=0.7, means=None):
        super().__init__(nb_arm, range=range, means=means)

    def pull(self, arm):
        if np.random.rand() < self.means[arm]:
            return 1.0
        else:
            return 0.0

#-----------------------------------------------------------------------
# Main Experiment Runner
#-----------------------------------------------------------------------

def experiment(
    algo,
    bandits,
    satisfaction_level,
    nb_step,
    nb_repetition=1,
    parameter=1,
    error_type="quantile",
):
    """
    Run a sequence of experiments using the specified algorithm and return averaged metrics.
    
    Parameters:
    - algo: Algorithm (function) used for the experiments
    - bandits: A list of bandit instances on which to run the algorithm
    - satisfaction_level: Satisfaction level
    - nb_step: Duration of each run
    - nb_repetition: Number of repetitions of the experiment on each bandit instance
    - parameter: Optional parameter for the algorithm
    - error_type: Type of error bar to use ("std" or "quantile")
    
    Returns:
    - Tuple of (average rewards, average regrets, average expectations, error bars)
    """
    if seed is not None:
        np.random.seed(seed)
    print(f"Experiment ({nb_repetition} runs)")
    rewards, regrets, expectations = [], [], []
    
    # Set initial prior variance (used in some Bayesian algorithms)
    setting.val_prior = 0
    
    for bandit in bandits:  # Average over multiple bandit instances
        for _ in progressbar(range(nb_repetition)):  # Repeat experiment on each bandit
            setting.val_prior += 1
            rew, reg, exp = algo(
                bandit, satisfaction_level, nb_step, parameter=parameter
            )
            rewards.append(rew)
            regrets.append(reg)
            expectations.append(exp)
    
    # Calculate error bars based on specified type
    if error_type == "std":
        error_bar_figure = np.std(np.cumsum(regrets, axis=1), axis=0)
    elif error_type == "quantile":
        error_bar_figure = np.quantile(
            np.cumsum(regrets, axis=1), q=[0.25, 0.75], axis=0
        )

    return (
        np.mean(rewards, 0),
        np.mean(regrets, 0),
        np.mean(expectations, 0),
        error_bar_figure,
    )

#-----------------------------------------------------------------------
# Core Algorithms from the Paper
#-----------------------------------------------------------------------

def ucb(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Implementation of UCB1 algorithm.
    
    The optional parameter corresponds to a scaling factor for the confidence intervals.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Use UCB1 strategy for remaining steps
    for i in range(nb_arm, nb_step):
        # Compute confidence intervals and indices
        R = confidence_multiplier * np.sqrt(2 * np.log(1 + i * (np.log(i) ** 2)))
        confidence = np.ones(nb_arm) * R / np.sqrt(nb_pull)
        emp_avg = arm_rewards / nb_pull
        ucb = emp_avg + confidence

        chosen_arm = np.argmax(ucb)

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def simple_sat(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Simple-Sat (Algorithm 1): Simple algorithm for Satisficing in the Realizable Case.
    
    Exploits if the empirical best arm exceeds the satisfaction level,
    and explores uniformly at random otherwise.
    
    Returns rewards, regrets, and expected rewards.
    """
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Simple-Sat strategy for remaining steps
    for i in range(nb_arm, nb_step):
        emp_avg = arm_rewards / nb_pull
        best_arm = np.argmax(emp_avg)
        if emp_avg[best_arm] >= satisfaction_level:
            chosen_arm = best_arm
        else:
            chosen_arm = np.random.randint(nb_arm)

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def sat_ucb(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Sat-UCB (Algorithm 2): Algorithm for Satisficing in the General Case.
    
    Uses confidence bounds to make decisions, achieving constant regret
    in the realizable case and logarithmic regret in the non-realizable case.
    
    The optional parameter corresponds to a scaling factor for the confidence intervals.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Sat-UCB strategy for remaining steps
    for i in range(nb_arm, nb_step):
        # Compute confidence intervals and indices
        R = confidence_multiplier * np.sqrt(2 * np.log(1 + i * (np.log(i) ** 2)))
        confidence = np.ones(nb_arm) * R / np.sqrt(nb_pull)
        emp_avg = arm_rewards / nb_pull
        ucb = emp_avg + confidence
        lcb = emp_avg - confidence
        ratio = (ucb - np.maximum(lcb, satisfaction_level)) / confidence

        max_ucb_arm = np.argmax(ucb)
        max_avg_arm = np.argmax(emp_avg)
        if emp_avg[max_avg_arm] >= satisfaction_level:
            chosen_arm = np.argmax(ratio)
        elif ucb[max_ucb_arm] >= satisfaction_level:
            indices = np.where(ucb >= satisfaction_level)[0]
            chosen_arm = np.random.choice(indices)
        else:
            chosen_arm = max_ucb_arm

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def sat_ucb_plus(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Sat-UCB+ (Algorithm 3): Modified version of Sat-UCB without random exploration.
    
    Similar to Sat-UCB but always chooses arm based on confidence bounds,
    never using uniform random selection among promising arms.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm
    
    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    
    # Sat-UCB+ strategy for remaining steps
    for i in range(nb_arm, nb_step):
        # Compute confidence intervals and indices
        R = confidence_multiplier * np.sqrt(2 * np.log(1 + i * (np.log(i) ** 2)))
        confidence = np.ones(nb_arm) * R / np.sqrt(nb_pull)
        emp_avg = arm_rewards / nb_pull
        ucb = emp_avg + confidence
        lcb = emp_avg - confidence
        ratio = (ucb - np.maximum(lcb, satisfaction_level)) / confidence
        max_ucb_arm = np.argmax(ucb)
        if ucb[max_ucb_arm] >= satisfaction_level:
            chosen_arm = np.argmax(ratio)
        else:
            chosen_arm = max_ucb_arm
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


#-----------------------------------------------------------------------
# Variants of Sat-UCB
#-----------------------------------------------------------------------

def sat_ucb_x_ucb(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Sat-UCB x UCB1: Variant of Algorithm 3 using UCB instead of ratio index.
    
    Uses the UCB index instead of the ratio index for exploitation.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Algorithm 3 x UCB1 strategy for remaining steps
    for i in range(nb_arm, nb_step):
        # Compute confidence intervals and indices
        R = confidence_multiplier * np.sqrt(2 * np.log(1 + i * (np.log(i) ** 2)))
        confidence = np.ones(nb_arm) * R / np.sqrt(nb_pull)
        emp_avg = arm_rewards / nb_pull
        ucb = emp_avg + confidence

        max_ucb_arm = np.argmax(ucb)
        m = satisfaction_level
        mi = -1
        # Find arm with highest UCB among those with average reward > satisfaction level
        for i in range(ucb.shape[0]):
            if ucb[i] >= m and emp_avg[i] >= satisfaction_level:
                m = ucb[i]
                mi = i
        if mi != -1:
            chosen_arm = mi
        elif ucb[max_ucb_arm] >= satisfaction_level:
            indices = np.where(ucb >= satisfaction_level)[0]
            chosen_arm = np.random.choice(indices)
        else:
            chosen_arm = max_ucb_arm

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def sat_ucb_x_avg(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Sat-UCB x Average: Variant of Algorithm 3 using empirical average reward.
    
    Uses empirical average reward instead of ratio index for exploitation.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Algorithm 3 x Average strategy for remaining steps
    for i in range(nb_arm, nb_step):
        # Compute confidence intervals and indices
        R = confidence_multiplier * np.sqrt(2 * np.log(1 + i * (np.log(i) ** 2)))
        confidence = np.ones(nb_arm) * R / np.sqrt(nb_pull)
        emp_avg = arm_rewards / nb_pull
        ucb = emp_avg + confidence

        max_ucb_arm = np.argmax(ucb)
        max_avg_arm = np.argmax(emp_avg)
        if emp_avg[max_avg_arm] >= satisfaction_level:
            chosen_arm = np.argmax(emp_avg)
        elif ucb[max_ucb_arm] >= satisfaction_level:
            indices = np.where(ucb >= satisfaction_level)[0]
            chosen_arm = np.random.choice(indices)
        else:
            chosen_arm = max_ucb_arm

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


#-----------------------------------------------------------------------
# Alternative Algorithms for Comparison
#-----------------------------------------------------------------------

def bernoulli_thompson(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Implementation of Thompson Sampling for Bernoulli bandits.
    
    Returns rewards, regrets, and expected rewards.
    """
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm
    setting.alpha = np.ones(nb_arm)
    beta = np.ones(nb_arm)

    for i in range(nb_step):
        theta = np.random.beta(setting.alpha, beta)
        chosen_arm = np.argmax(theta)

        # Get reward
        reward = bandit.pull(chosen_arm)
        # Update distributions
        setting.alpha[chosen_arm] += reward
        beta[chosen_arm] += 1 - reward
        # Update metrics
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def epsilon_greedy(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Implementation of epsilon-greedy algorithm.
    
    Uses epsilon_n = c*K/(n^2) where K is the number of arms.
    The parameter c is specified by the parameter argument.
    
    Returns rewards, regrets, and expected rewards.
    """
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    epsilon = 1
    c = parameter
    d = 1
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    for i in range(nb_step):
        epsilon = min(1, c * nb_arm / (d * d * (i + 1)))
        if np.random.uniform(0, 1) > epsilon:
            chosen_arm = np.argmax(arm_rewards / np.maximum(nb_pull, 1))
        else:
            chosen_arm = np.random.randint(0, nb_arm)

        # Get reward
        reward = bandit.pull(chosen_arm)
        # Update metrics
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def simple_sat_round_robin(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Simple-Sat with Round Robin: Variant of Simple-Sat using round-robin exploration.
    
    Returns rewards, regrets, and expected rewards.
    """
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    do_rr = True
    rr_cur = 0
    for i in range(nb_arm, nb_step):
        emp_avg = arm_rewards / nb_pull
        best_arm = np.argmax(emp_avg)
        if emp_avg[best_arm] >= satisfaction_level:
            chosen_arm = best_arm
        else:
            do_rr = True

        if do_rr:
            chosen_arm = rr_cur
            rr_cur = rr_cur + 1
            if rr_cur >= nb_arm:
                do_rr = False
                rr_cur = 0
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations


def ucb_alpha_elimination(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Implementation of UCB-alpha with arm elimination.
    
    Eliminates arms whose upper confidence bounds are below the 
    lower confidence bound of the current best arm.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm

    # Play each arm twice
    for i in range(2 * nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i % nb_arm
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Calculate confidence radii for each arm
    epsilon_arm = np.sqrt(
        (2 / nb_pull) * (np.log((3 * np.log(nb_pull) ** 2 / setting.delta_conf_level)))
    )

    for i in range(2 * nb_arm, nb_step):
        # Compute confidence intervals and indices
        chosen_arm = np.argmax(
            (arm_rewards / nb_pull)
            + (confidence_multiplier * setting.alpha * epsilon_arm)
        )
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        epsilon_arm[chosen_arm] = np.sqrt(
            (2 / nb_pull[chosen_arm])
            * (
                np.log(
                    (3 * np.log(nb_pull[chosen_arm]) ** 2 / setting.delta_conf_level)
                )
            )
        )
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
        
        # Eliminate bad arms by setting their rewards and epsilon_arms to zero
        chosen_arm = np.argmax(arm_rewards / nb_pull)
        plus_val = (arm_rewards / nb_pull) + epsilon_arm
        arm_rewards[
            plus_val
            < (arm_rewards[chosen_arm] / nb_pull[chosen_arm]) - epsilon_arm[chosen_arm]
        ] = 0
        epsilon_arm[
            plus_val
            < (arm_rewards[chosen_arm] / nb_pull[chosen_arm]) - epsilon_arm[chosen_arm]
        ] = 0

    return rewards, regrets, expectations


def satisfaction_mean_reward(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Implementation of Satisfaction in Mean Reward UCL.
    
    Uses Bayesian confidence bounds to select arms above the satisfaction level.
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm
    
    # Parameters for Bayesian algorithm
    sigma_s = 1  # True variance
    mu_0 = 0.5  # Mean of prior for each arm
    kappa = np.sqrt(2 * np.pi * np.e)  # Constant from the paper
    delta_var = 0
    
    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Function to evaluate UCL for each arm
    def eval_q_arm(arm_rewards, i):
        q_arm = (
            ((delta_var**2) * mu_0 + (arm_rewards)) / ((delta_var**2) + nb_pull)
        ) + confidence_multiplier * (
            sigma_s / np.sqrt((delta_var**2) + nb_pull)
        ) * norm.ppf(
            1 - 1 / (kappa * (i)), loc=0, scale=1
        )
        return q_arm

    # Process all arms in the eligible set each step
    count = nb_arm - 1
    step_error = 1
    while count < nb_step - 1:
        q_arm = eval_q_arm(arm_rewards, step_error)
        if np.where(q_arm > satisfaction_level)[0].size > 0:
            index = 0
            while count < nb_step - 1 and index < len(
                np.where(q_arm > satisfaction_level)[0]
            ):
                chosen_arm = np.where(q_arm > satisfaction_level)[0][index]
                reward = bandit.pull(chosen_arm)
                nb_pull[chosen_arm] += 1
                arm_rewards[chosen_arm] += reward
                rewards.append(reward)
                regrets.append(
                    max(
                        0,
                        min(satisfaction_level, best_arm_expectation)
                        - bandit.means[chosen_arm],
                    )
                )
                expectations.append(bandit.means[chosen_arm])
                count += 1
                index += 1
        else:
            chosen_arm = np.argmax(q_arm)
            reward = bandit.pull(chosen_arm)
            nb_pull[chosen_arm] += 1
            arm_rewards[chosen_arm] += reward
            rewards.append(reward)
            regrets.append(
                max(
                    0,
                    min(satisfaction_level, best_arm_expectation)
                    - bandit.means[chosen_arm],
                )
            )
            expectations.append(bandit.means[chosen_arm])
            count += 1
        # Increase the step of the error
        step_error += 1
    return rewards, regrets, expectations


def deterministic_ucl(bandit, satisfaction_level, nb_step, parameter=1):
    """
    Implementation of Deterministic UCL (Bayes-UCB).
    
    Returns rewards, regrets, and expected rewards.
    """
    confidence_multiplier = parameter
    nb_arm = bandit.nb_arm
    best_arm_expectation = np.max(bandit.means)
    rewards = []  # Empirical reward at each step
    regrets = []  # Satisfying regret at each step
    expectations = []  # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm)  # Number of pulls of each arm
    arm_rewards = np.zeros(nb_arm)  # Empirical total reward of each arm
    
    # Parameters for Bayesian algorithm
    sigma_s = 1  # True variance
    simga_0 = setting.val_prior  # Standard deviation of prior for each arm
    mu_0 = 0.5  # Mean of prior for each arm
    kappa = np.sqrt(2 * np.pi * np.e)  # Constant from the paper
    delta_var = 0
    
    # Play each arm once
    for i in range(nb_arm):
        if i >= nb_step:
            break
        chosen_arm = i
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])

    # Function to evaluate UCL for each arm
    def eval_q_arm(arm_rewards, i):
        q_arm = (
            ((delta_var**2) * mu_0 + arm_rewards) / ((delta_var**2) + nb_pull)
        ) + confidence_multiplier * (
            sigma_s / np.sqrt((delta_var**2) + nb_pull)
        ) * norm.ppf(
            1 - 1 / (kappa * (i)), loc=0, scale=1
        )
        return q_arm

    # Always choose arm with highest UCL
    for i in range(nb_arm + 1, nb_step + 1):
        q_arm = eval_q_arm(arm_rewards, i - nb_arm)
        chosen_arm = np.argmax(q_arm)
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(
            max(
                0,
                min(satisfaction_level, best_arm_expectation)
                - bandit.means[chosen_arm],
            )
        )
        expectations.append(bandit.means[chosen_arm])
    return rewards, regrets, expectations