import numpy as np
import matplotlib.pyplot as plt
import sys

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

class Bandit():
    def __init__(self, nb_arm, range = [0, 1], threshold = 0.7,means=None):
        self.nb_arm = nb_arm
        if means == None:
            self.means = np.random.rand(nb_arm)*(range[1]-range[0]) + range[0]
            self.means[0] = np.random.rand()*(range[1]-threshold) + threshold
        else:
            self.means = means
        #print(self.means)
    
    def pull(self,arm):
        return self.means[arm]

class GaussianBandit(Bandit):
    def __init__(self, nb_arm, range = [0, 1], threshold = 0.7, means=None, sigmas = None):
        super().__init__(nb_arm,range=range,threshold=threshold,means=means)
        if sigmas != None:
            self.sigmas = sigmas
        else:
            self.sigmas = np.ones(nb_arm)
    
    def pull(self,arm):
        return np.random.randn()*self.sigmas[arm]+self.means[arm]

def ucb(bandit,satisfaction_level,nb_step):
    nb_arm = bandit.nb_arm
    rewards = [] # Empirical reward at each step
    regrets = [] # Satisfying regret at each step
    expectations = [] # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm) # Number of pull of each arm
    arm_rewards = np.zeros(nb_arm) # Empirical total reward of each arm

    for i in range(nb_arm): # Play each arm once
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])

    for i in range(nb_arm,nb_step):
        # Compute confidences interval and indices
        R = np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        confidence = np.ones(nb_arm)*R / np.sqrt(nb_pull)
        emp_avg = arm_rewards/nb_pull
        ucb = emp_avg + confidence

        chosen_arm = np.argmax(ucb)

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations

def algo1(bandit,satisfaction_level,nb_step):
    nb_arm = bandit.nb_arm
    rewards = [] # Empirical reward at each step
    regrets = [] # Satisfying regret at each step
    expectations = [] # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm) # Number of pull of each arm
    arm_rewards = np.zeros(nb_arm) # Empirical total reward of each arm

    for i in range(nb_arm): # Play each arm once
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])

    for i in range(nb_arm,nb_step):
        # Compute confidences interval and indices
        R = np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        confidence = np.ones(nb_arm)*R / np.sqrt(nb_pull)
        emp_avg = arm_rewards/nb_pull
        ucb = emp_avg + confidence
        lcb = emp_avg - confidence
        ratio = (ucb - np.max(lcb,satisfaction_level))/confidence

        max_avg_arm = np.argmax(emp_avg)
        if emp_avg[max_avg_arm] >= satisfaction_level:
            chosen_arm = max_avg_arm
        else:
            chosen_arm = np.random.randint(0,nb_arm)

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations



def algo3(bandit,satisfaction_level,nb_step):
    nb_arm = bandit.nb_arm
    rewards = [] # Empirical reward at each step
    regrets = [] # Satisfying regret at each step
    expectations = [] # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm) # Number of pull of each arm
    arm_rewards = np.zeros(nb_arm) # Empirical total reward of each arm

    for i in range(nb_arm): # Play each arm once
        if i >= nb_step:
            break
        chosen_arm = i
        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])

    for i in range(nb_arm,nb_step):
        # Compute confidences interval and indices
        R = np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        confidence = np.ones(nb_arm)*R / np.sqrt(nb_pull)
        emp_avg = arm_rewards/nb_pull
        ucb = emp_avg + confidence
        lcb = emp_avg - confidence
        ratio = (ucb - np.maximum(lcb,satisfaction_level))/confidence

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
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations


def experiment(algo,bandits,satisfaction_level,nb_step,nb_repetition=1):
    print(f"Experiment ({nb_repetition} runs)")
    rewards,regrets,expectations= [],[],[]
    for bandit in bandits:  # In case we want to average over multiple bandit instances
        for _ in progressbar(range(nb_repetition)):  # Repetition of the experiment on the bandit instance
            rew,reg,exp = algo(bandit, satisfaction_level, nb_step)
            rewards.append(rew)
            regrets.append(reg)
            expectations.append(exp)
    return np.mean(rewards,0),np.mean(regrets,0),np.mean(expectations,0)   # Averaging the results


def compute_bound_algo3_sat(bandit,satisfaction_level): # Bound on satisficing regret of algo3 assuming there exist a satisfying arm
    R = 0
    delta_star = np.max(bandit.means) - satisfaction_level
    print(delta_star)
    for m in bandit.means:
        if m<satisfaction_level:
            delta = satisfaction_level - m
            R += delta + 2/delta + 7*delta/(delta_star**2)
    return R


satisfaction_level = 0.8
nb_arm = 20
nb_step = 1000
nb_bandit = 1


bandits = [GaussianBandit(nb_arm,means=[i/nb_arm for i in range(nb_arm)],sigmas=[1 for i in range(nb_arm)]) for _ in range(nb_bandit)]

# Experiment with 1 run of the algorithms
nb_repetition = 1
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
print("Number of non-satisfying actions played : ",np.sum(expectations_ucb<satisfaction_level))

rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)
print("Number of non-satisfying actions played : ",np.sum(expectations_algo3<satisfaction_level))

plt.plot(expectations_ucb, '.',label="UCB1")
plt.plot(expectations_algo3, '.', label="Algorithm 3")
plt.plot(np.ones_like(expectations_algo3)*satisfaction_level,label="Satisfaction level")
plt.title("Expected reward of the arm played at each step")
plt.xlabel("Time step")
plt.ylabel("Expected reward")
plt.legend()

plt.figure()
plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
plt.title("Satisficing regret (1 run)")
plt.xlabel("Time step")
plt.ylabel("Statisficing regret")
plt.legend()

plt.show()

# Experiment in the realizable case
print("Realizable case")
nb_step = 10000
nb_repetition = 50
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)

plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
# Plot bound ?
# plt.plot(np.ones_like(regrets_algo3)*compute_bound_algo3_sat(bandits[0], satisfaction_level),label="Regret bound for Algorithm 3")
plt.title("Average satisficing regret over 50 runs\nRealizable case")
plt.xlabel("Time step")
plt.ylabel("Statisficing regret")
plt.legend()

plt.show()

# Experiment in the unrealizable case
print("Unrealizable case")
satisfaction_level = 1
rewards_ucb,regrets_ucb,expectations_ucb = experiment(ucb, bandits, satisfaction_level, nb_step, nb_repetition)
rewards_algo3,regrets_algo3,expectations_algo3 = experiment(algo3,bandits, satisfaction_level, nb_step, nb_repetition)

plt.plot(np.cumsum(regrets_ucb),label="UCB1")
plt.plot(np.cumsum(regrets_algo3), label="Algorithm 3")
plt.title("Average satisficing regret over 50 runs\nUnrealizable case")
plt.xlabel("Time step")
plt.ylabel("Statisficing regret")
plt.legend()


plt.show()