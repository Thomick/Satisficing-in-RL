from bandit import *

satisfaction_level = 0.8
nb_arm = 20
nb_step = 10000
nb_bandit = 1
nb_repetition = 50


bandits = [GaussianBandit(nb_arm,means=[i/nb_arm for i in range(nb_arm)],sigmas=[1 for i in range(nb_arm)]) for _ in range(nb_bandit)]

# Experiment with 1 run of the algorithms
nb_repetition = 1
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
print("Number of non-satisfying actions played : ",np.sum(expectations_ucb<satisfaction_level))

rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)
print("Number of non-satisfying actions played : ",np.sum(expectations_algo3<satisfaction_level))

plt.figure(figsize=(8, 6))
plt.plot(expectations_ucb, '.',label="UCB1")
plt.plot(expectations_algo3, '.', label="Algorithm 3")
plt.plot(np.ones_like(expectations_algo3)*satisfaction_level,label="Satisfaction level")
plt.title("Expected reward of the arm played at each step\nGaussian rewards")
plt.xlabel("Time step")
plt.ylabel("Expected reward")
plt.legend()

plt.tight_layout()
plt.figure(figsize=(8, 6))
plt.tight_layout()
plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
rewards_algo3xucb,regrets_algo3xucb,expectations_algo3xucb = experiment(algo3xucb,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xucb), label="Algorithm 3 x UCB1")
rewards_algo3xavg,regrets_algo3xavg,expectations_algo3xavg = experiment(algo3xavg,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xavg), label="Algorithm 3 x Average reward")
plt.title("Satisficing regret (1 run) - Gaussian rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()

plt.tight_layout()
plt.figure(figsize=(8, 6))

# Experiment in the realizable case
print("Realizable case")
nb_step = 10000
nb_repetition = 50
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)

plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
rewards_algo3xucb,regrets_algo3xucb,expectations_algo3xucb = experiment(algo3xucb,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xucb), label="Algorithm 3 x UCB1")
rewards_algo3xavg,regrets_algo3xavg,expectations_algo3xavg = experiment(algo3xavg,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xavg), label="Algorithm 3 x Average reward")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.5)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.5)")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.25)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.25)")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.125)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.125)")
# Plot bound ?
# plt.plot(np.ones_like(regrets_algo3)*compute_bound_algo3_sat(bandits[0], satisfaction_level),label="Regret bound for Algorithm 3")
plt.title("Average satisficing regret over 50 runs\nRealizable case - Gaussian rewards")
plt.xlabel("Time step")
plt.ylabel("Statisficing regret")
plt.legend()

plt.tight_layout()


plt.figure(figsize=(8, 6))
# Experiment in the unrealizable case
print("Unrealizable case")
satisfaction_level = 1
nb_step = 10000
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)

plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
rewards_algo3xucb,regrets_algo3xucb,expectations_algo3xucb = experiment(algo3xucb,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xucb), label="Algorithm 3 x UCB1")
rewards_algo3xavg,regrets_algo3xavg,expectations_algo3xavg = experiment(algo3xavg,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xavg), label="Algorithm 3 x Average reward")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.5)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.5)")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.25)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.25)")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.125)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.125)")
plt.title("Average satisficing regret over 50 runs\nUnrealizable case - Gaussian rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
nb_step =1000
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, 0.25, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=0.25$")
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, 0.5, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=0.5$")
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, 0.75, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=0.75$")
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, np.max(bandits[0].means), nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=\rho^*$")
plt.title("Average satisficing regret over 50 runs\nAlgorithm 3 - Gaussian rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()


plt.tight_layout()
plt.show()