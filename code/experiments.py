# Experiments for "Regret Bounds for Satisficing in Multi-Armed Bandit Problems"
#
# This script reproduces the experiments from the paper.
# Configure the parameters below to run different experiments.

import setting
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Initialize global settings
setting.init()
from bandit import *

# Configure matplotlib
plt.rcParams.update(
    {
        "font.size": 15,
        "text.usetex": False,  # Set to True if you have LaTeX installed
    }
)

#-----------------------------------------------------------------------
# Experiment Configuration
#-----------------------------------------------------------------------

# Toggle which algorithms to include
plot_satucb = True           # Enable Sat-UCB algorithm
plot_satucbplus = True       # Enable Sat-UCB+ algorithm
plot_satucb_variants = False # Enable Sat-UCB variants
plot_other_algorithms = True # Enable comparison algorithms
plot_ucb_algorithm = True    # Enable UCB1 algorithm

# Toggle special experiments
experiment_multiple_satisfaction_levels = False # Compare different satisfaction levels
experiment_1_run = False                        # Run a single simulation
experiment_round_robin = False                  # Test round-robin exploration

# Choose experimental setting
reward_distribution = "gaussian"  # "gaussian" or "bernoulli"
realizable_case = True            # True for realizable case, False for non-realizable
setting_id = 1                    # 1, 2 or 3 (different arm distributions)

# General experiment parameters
nb_arm = 20                  # Number of arms
nb_repetition = 50           # Number of repetitions for each algorithm
resolution_value = 300       # DPI for saving figures

# Plot parameters
plot_error = True            # Show error bands/bars
error_type = "std"           # "std" or "quantile"
error_bar = True             # True for error bars, False for error bands

#-----------------------------------------------------------------------
# Experiment Setup
#-----------------------------------------------------------------------

# Configure experiment based on case (realizable or not)
if realizable_case:
    nb_step = 10000
    errorevery_bar = 3300
else:
    nb_step = 50000
    errorevery_bar = 12000

# Set up bandit means based on setting
if setting_id == 1:
    # Setting 1: Arms with evenly spaced means from 0 to 1
    mean_arms = [i / nb_arm for i in range(nb_arm)]
    satisfaction_level = 0.8 if realizable_case else 1.0
elif setting_id == 2:
    # Setting 2: Arms with means distributed as sqrt(i/n)
    mean_arms = [(i / nb_arm) ** 0.5 for i in range(nb_arm)]
    mean_arms.reverse()  # Reversed so best arm is not the last one
    satisfaction_level = 0.92 if realizable_case else 1.1
elif setting_id == 3:
    # Setting 3: One arm with mean 1, all others with mean 0
    mean_arms = [0 for i in range(nb_arm)]
    mean_arms[0] = 1.0
    satisfaction_level = 0.5 if realizable_case else 1.1
else:
    raise ValueError("Undefined setting_id")

# Print experiment configuration details
print("The mean of arms are: ", mean_arms)
copy_mean_arms = np.array(mean_arms)
copy_mean_arms.sort()
maximum_arm = np.max(mean_arms)
print("Arm means:", copy_mean_arms)
print("The largest mean of the arms is: ", maximum_arm)
print("The second largest mean of the arms is: ", copy_mean_arms[-2])
print("The satisfying level is: ", satisfaction_level)
print(
    "The number of satisfying arms is ",
    len(copy_mean_arms[copy_mean_arms >= satisfaction_level]),
)

# Create bandit instances
if reward_distribution == "gaussian":
    bandits = [
        GaussianBandit(
            nb_arm,
            means=[mean_arms[i] for i in range(nb_arm)],
            sigmas=[1 for i in range(nb_arm)],
        )
    ]
elif reward_distribution == "bernoulli":
    bandits = [
        BernoulliBandit(
            nb_arm,
            means=[mean_arms[i] for i in range(nb_arm)],
        )
    ]

#-----------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------

def plot_results(algo_name, regrets, var, plot_error, error_bar, errorevery_bar=0):
    """Plot the results for a given algorithm with error bars/bands."""
    if not plot_error:
        plt.plot(np.cumsum(regrets), label=algo_name)
    else:
        cum_regrets = np.cumsum(regrets)
        if not error_bar:
            # Plot error bands
            plt.plot(cum_regrets, label=algo_name)
            if var.shape[0] == 2:
                plt.fill_between(
                    np.arange(nb_step),
                    var[0],
                    var[1],
                    alpha=0.2,
                    label="_{}".format(algo_name),
                )
            else:
                var = var / np.sqrt(nb_repetition)
                plt.fill_between(
                    np.arange(nb_step),
                    cum_regrets - var,
                    cum_regrets + var,
                    alpha=0.2,
                    label="_{}".format(algo_name),
                )
        else:
            # Plot error bars
            if var.shape[0] == 2:
                plt.errorbar(
                    np.arange(nb_step),
                    cum_regrets,
                    yerr=[
                        np.maximum(cum_regrets - var[0], 0),
                        np.maximum(var[1] - cum_regrets, 0),
                    ],
                    errorevery=errorevery_bar,
                    label=algo_name,
                )
            else:
                var = var / np.sqrt(nb_repetition)
                plt.errorbar(
                    np.arange(nb_step),
                    cum_regrets,
                    yerr=var,
                    errorevery=errorevery_bar,
                    label=algo_name,
                )

#-----------------------------------------------------------------------
# Run Experiments
#-----------------------------------------------------------------------

# Set up the plot
plt.figure(figsize=(8, 6))

# Experiment 1: Single Run
if experiment_1_run:
    rewards_ucb, regrets_ucb, expectations_ucb, std_ucb = experiment(
        ucb,
        bandits,
        satisfaction_level,
        1000,
        1,
        error_type=error_type,
    )
    rewards_sat_ucb, regrets_sat_ucb, expectations_sat_ucb, std_sat_ucb = experiment(
        sat_ucb,
        bandits,
        satisfaction_level,
        1000,
        1,
        error_type=error_type,
    )
    plt.plot(expectations_ucb, ".", label="UCB1")
    plt.plot(expectations_sat_ucb, ".", label="Sat-UCB")
    plt.plot(
        np.ones_like(expectations_sat_ucb) * satisfaction_level,
        label="Satisfaction level",
    )
    plt.xlabel("Time step")
    plt.ylabel("Expected reward")
    plt.legend()
    plt.tight_layout()
    plt.show()
    exit()

# Experiment 2: Round Robin
if experiment_round_robin:
    errorevery_bar -= 200
    rewards_simple_sat, regrets_simple_sat, expectations_simple_sat, std_simple_sat = experiment(
        simple_sat,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    plot_results(
        "Simple-Sat",
        regrets_simple_sat,
        std_simple_sat,
        plot_error,
        error_bar,
        errorevery_bar,
    )

    errorevery_bar -= 200
    rewards_simple_sat_rr, regrets_simple_sat_rr, expectations_simple_sat_rr, std_simple_sat_rr = experiment(
        simple_sat_round_robin,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    plot_results(
        "Simple-Sat with Round Robin",
        regrets_simple_sat_rr,
        std_simple_sat_rr,
        plot_error,
        error_bar,
        errorevery_bar,
    )

# Experiment 3: Sat-UCB (Algorithm 2)
if plot_satucb:
    errorevery_bar -= 200
    rewards_sat_ucb, regrets_sat_ucb, expectations_sat_ucb, std_sat_ucb = experiment(
        sat_ucb,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    plot_results(
        "Sat-UCB",
        regrets_sat_ucb,
        std_sat_ucb,
        plot_error,
        error_bar,
        errorevery_bar,
    )

# Experiment 4: Sat-UCB+ (Algorithm 3)
if plot_satucbplus:
    errorevery_bar -= 200
    rewards_sat_ucb_plus, regrets_sat_ucb_plus, expectations_sat_ucb_plus, std_sat_ucb_plus = experiment(
        sat_ucb_plus,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    plot_results(
        "Sat-UCB+",
        regrets_sat_ucb_plus,
        std_sat_ucb_plus,
        plot_error,
        error_bar,
        errorevery_bar,
    )

# Experiment 5: Multiple Satisfaction Levels
if experiment_multiple_satisfaction_levels:
    if setting_id == 3:
        satisfaction_level_list = [1.0, 1.025, 1.05, 1.1, 1.2]
    elif setting_id == 1:
        satisfaction_level_list = [0.95, 0.96, 0.97, 1.0, 1.1]
    
    for level in satisfaction_level_list:
        rewards_sat_ucb_plus, regrets_sat_ucb_plus, expectations_sat_ucb_plus, std_sat_ucb_plus = experiment(
            sat_ucb_plus,
            bandits,
            level,
            nb_step,
            nb_repetition,
            error_type=error_type,
        )
        errorevery_bar -= 200
        plot_results(
            f"Sat-UCB+ S-Level: {level}",
            regrets_sat_ucb_plus,
            std_sat_ucb_plus,
            plot_error,
            error_bar,
            errorevery_bar,
        )

# Experiment 6: Sat-UCB Variants
if plot_satucb_variants:
    # Sat-UCB x UCB1
    rewards_sat_ucb_x_ucb, regrets_sat_ucb_x_ucb, expectations_sat_ucb_x_ucb, std_sat_ucb_x_ucb = experiment(
        sat_ucb_x_ucb,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    errorevery_bar -= 200
    plot_results(
        "Sat-UCB x UCB1",
        regrets_sat_ucb_x_ucb,
        std_sat_ucb_x_ucb,
        plot_error,
        error_bar,
        errorevery_bar,
    )
    
    # Sat-UCB x Average
    rewards_sat_ucb_x_avg, regrets_sat_ucb_x_avg, expectations_sat_ucb_x_avg, std_sat_ucb_x_avg = experiment(
        sat_ucb_x_avg,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    errorevery_bar -= 200
    plot_results(
        "Sat-UCB x Average",
        regrets_sat_ucb_x_avg,
        std_sat_ucb_x_avg,
        plot_error,
        error_bar,
        errorevery_bar,
    )

# Experiment 7: UCB1
if plot_ucb_algorithm:
    rewards_ucb, regrets_ucb, expectations_ucb, std_ucb = experiment(
        ucb, 
        bandits, 
        satisfaction_level, 
        nb_step, 
        nb_repetition, 
        error_type=error_type
    )
    errorevery_bar -= 200
    plot_results(
        "UCB1", 
        regrets_ucb, 
        std_ucb, 
        plot_error, 
        error_bar, 
        errorevery_bar
    )

# Experiment 8: Other Algorithms
if plot_other_algorithms:
    # UCB-alpha
    setting.alpha = 1
    setting.delta_conf_level = 0.001
    rewards_ucb_alpha, regrets_ucb_alpha, expectations_ucb_alpha, std_ucb_alpha = experiment(
        ucb_alpha_elimination,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    errorevery_bar -= 200
    plot_results(
        "UCB-alpha",
        regrets_ucb_alpha,
        std_ucb_alpha,
        plot_error,
        error_bar,
        errorevery_bar,
    )
    
    # Satisfaction in Mean Reward UCL
    rewards_smr, regrets_smr, expectations_smr, std_smr = experiment(
        satisfaction_mean_reward,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    errorevery_bar -= 200
    plot_results(
        "Satisfaction in Mean Reward UCL",
        regrets_smr,
        std_smr,
        plot_error,
        error_bar,
        errorevery_bar,
    )
    
    # Deterministic UCL
    rewards_ducl, regrets_ducl, expectations_ducl, std_ducl = experiment(
        deterministic_ucl,
        bandits,
        satisfaction_level,
        nb_step,
        nb_repetition,
        error_type=error_type,
    )
    errorevery_bar -= 200
    plot_results(
        "Deterministic UCL",
        regrets_ducl,
        std_ducl,
        plot_error,
        error_bar,
        errorevery_bar,
    )

#-----------------------------------------------------------------------
# Plot Formatting and Saving
#-----------------------------------------------------------------------

# Set different line styles for better visibility
linestyles = [
    "solid",
    "dotted",
    "dashed",
    "dashdot",
    (0, (3, 1, 1, 1, 1, 1)),
    (5, (10, 3)),
    (0, (3, 1, 1, 1)),
]

ax = plt.gca()
for l, ls in zip(ax.lines, itertools.cycle(linestyles)):
    l.set_linestyle(ls)

# Set plot title and labels
reward_type_name = "Gaussian" if reward_distribution == "gaussian" else "Bernoulli"

if experiment_multiple_satisfaction_levels:
    plt.title(
        f"Average regret over {nb_repetition} runs\n{reward_type_name} rewards - Setting {setting_id}"
    )
    plt.xlabel("Time step")
    plt.ylabel("Regret")
elif realizable_case:
    plt.title(
        f"Average satisficing regret over {nb_repetition} runs\nRealizable case - {reward_type_name} rewards - Setting {setting_id}"
    )
    plt.xlabel("Time step")
    plt.ylabel("Satisficing regret")
else:
    plt.title(
        f"Average regret over {nb_repetition} runs\nNot realizable case - {reward_type_name} rewards - Setting {setting_id}"
    )
    plt.xlabel("Time step")
    plt.ylabel("Regret")

plt.legend(fontsize=12)
plt.tight_layout()

# Create directory for saving figures
os.makedirs("Figures/New", exist_ok=True)
os.makedirs(f"Figures/New/{reward_distribution}", exist_ok=True)

# Save the figure
if realizable_case:
    plt.savefig(
        f"Figures/New/{reward_distribution}/Realizable_Setting{setting_id}.png",
        format="png",
        dpi=resolution_value,
    )
else:
    plt.savefig(
        f"Figures/New/{reward_distribution}/NonRealizable_Setting{setting_id}.png",
        format="png",
        dpi=resolution_value,
    )

# Show the plot
plt.show()