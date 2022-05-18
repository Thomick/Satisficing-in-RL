from bandit import *

satisfaction_level = 0.8
nb_arm = 20
nb_step = 1000
nb_bandit = 1
nb_repetition = 50

# Bandit definition
bandits = [BernoulliBandit(nb_arm,means=[i/nb_arm for i in range(nb_arm)])]


######################## Experiment in the realizable case ########################
plt.figure(figsize=(8, 6))
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
#plt.title("Average satisficing regret over 50 runs\nRealizable case - Bernoulli rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()
plt.tight_layout()


######################## Experiment in the not realizable case ########################
print("Not realizable case")
plt.figure(figsize=(8, 6))

satisfaction_level = 1
nb_step = 50000

rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
rewards_algo3xucb,regrets_algo3xucb,expectations_algo3xucb = experiment(algo3xucb,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xucb), label="Algorithm 3 x UCB1")
rewards_algo3xavg,regrets_algo3xavg,expectations_algo3xavg = experiment(algo3xavg,bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3xavg), label="Algorithm 3 x Average reward")
#plt.title("Average satisficing regret over 50 runs\nNot realizable case - Bernoulli rewards")
plt.xlabel("Time step")
plt.ylabel("Regret")
plt.legend()

plt.tight_layout()

######################## Different satisfaction levels ########################
print("Difference depending on satisfaction level")
plt.figure(figsize=(8, 6))

nb_step = 10000

rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, 0.25, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=0.25$")
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, 0.5, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=0.5$")
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, 0.75, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=0.75$")
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, np.max(bandits[0].means), nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_algo3), label=r"$S=\rho^*$")
plt.title("Average satisficing regret over 50 runs\nAlgorithm 3 - Bernoulli rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()
plt.tight_layout()

######################## Comparison with other algorithms ########################


# Experiment with 20 arms
print("Realizable case")
plt.figure(figsize=(8, 6))

nb_step = 10000
nb_repetition = 50
c = 0.1

rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)

plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=0.5)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.5)")
rewards_greedy,regrets_greedy,expectations_greedy = experiment(epsilon_greedy, bandits, satisfaction_level, nb_step, nb_repetition,parameter=c)
plt.plot(np.cumsum(regrets_greedy), label="Epsilon greedy")
rewards_thompson,regrets_thompson,expectations_thompson = experiment(bernoulli_thompson, bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_thompson), label="Thompson sampling")
#plt.title("Average satisficing regret over 50 runs\nRealizable case - Bernoulli rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()
plt.tight_layout()



# Experiment with 200 arms
print("Realizable case")
plt.figure(figsize=(8, 6))

nb_step = 10000
nb_repetition = 50
c = 0.1
nb_arm = 200

bandits = [BernoulliBandit(nb_arm,means=[i/nb_arm for i in range(nb_arm)])]

rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)

plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition,parameter=c)
plt.plot(np.cumsum(regrets_ucb), label="UCB1 (confidence * 0.5)")
rewards_greedy,regrets_greedy,expectations_greedy = experiment(epsilon_greedy, bandits, satisfaction_level, nb_step, nb_repetition,parameter=epsilon)
plt.plot(np.cumsum(regrets_greedy), label="Epsilon greedy")
rewards_thompson,regrets_thompson,expectations_thompson = experiment(bernoulli_thompson, bandits, satisfaction_level, nb_step, nb_repetition)
plt.plot(np.cumsum(regrets_thompson), label="Thompson sampling")
#plt.title("Average satisficing regret over 50 runs\nRealizable case - Bernoulli rewards")
plt.xlabel("Time step")
plt.ylabel("Satisficing regret")
plt.legend()

plt.tight_layout()
plt.show()