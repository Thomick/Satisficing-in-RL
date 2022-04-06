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
    def __init__(self, nb_arm, range = [0, 1], threshold = 0.7):
        self.nb_arm = nb_arm
        self.means = np.random.rand(nb_arm)*(range[1]-range[0]) + range[0]
        self.means[0] = np.random.rand()*(range[1]-threshold) + threshold
        #print(self.means)
    
    def pull(self,arm):
        return self.means[arm]

class GaussianBandit(Bandit):
    def __init__(self, nb_arm, range = [0, 1], threshold = 0.7, sigmas = None):
        super().__init__(nb_arm)
        if sigmas != None:
            self.sigmas = sigmas
        else:
            self.sigmas = np.ones(nb_arm)
    
    def pull(self,arm):
        return np.random.randn()*self.sigmas[arm]+self.means[arm]

def ucb_suffice(bandit,satisfaction_level,nb_step):
    nb_arm = bandit.nb_arm
    rewards = []
    regrets = []
    expectations = []
    choices = [0 for _ in range(nb_arm)]
    nb_pull = [0 for _ in range(nb_arm)]
    arm_rewards = [0 for _ in range(nb_arm)]
    for i in range(1,nb_step+1):
        #print(f"Step {i}")
        R = np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        best_arm = 0
        satisfying_arm = None
        best_ucb = float('-inf')
        for j in range(nb_arm):
            if nb_pull[j] > 0:
                mean = arm_rewards[j]/nb_pull[j]
                if mean-R/np.sqrt(nb_pull[j]) >= satisfaction_level:
                    satisfying_arm = j
                    break
                ucb = mean+R/np.sqrt(nb_pull[j])
                if ucb > best_ucb:
                    best_arm = j
                    best_ucb = ucb
            else:
                best_arm = j
                best_ucb = float('inf')
        if satisfying_arm == None:
            chosen_arm = best_arm
            choices[chosen_arm] += 1
        else:
            chosen_arm = satisfying_arm
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations,choices

def cb_ratio(bandit,satisfaction_level,nb_step):
    nb_arm = bandit.nb_arm
    rewards = []
    regrets = []
    expectations = []
    choices = [0 for _ in range(nb_arm)]
    nb_pull = [0 for _ in range(nb_arm)]
    arm_rewards = [0 for _ in range(nb_arm)]
    for i in range(1,nb_step+1):
        #print(f"Step {i}")
        R = np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        best_arm = 0
        best_ratio = float('-inf')
        for j in range(nb_arm):
            if nb_pull[j] > 0:
                mean = arm_rewards[j]/nb_pull[j]
                ratio = (mean+R/np.sqrt(nb_pull[j])-satisfaction_level)/(2*R/np.sqrt(nb_pull[j]))
                if ratio > best_ratio:
                    best_arm = j
                    best_ratio = ratio
            else:
                best_arm = j
                best_ratio = float('inf')
        reward = bandit.pull(best_arm)
        nb_pull[best_arm] += 1
        arm_rewards[best_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[best_arm]))
        expectations.append(bandit.means[best_arm])
    return rewards,regrets,expectations,choices

def optimistic_mean_sat(bandit,satisfaction_level,nb_step):
    nb_arm = bandit.nb_arm
    rewards = []
    regrets = []
    expectations = []
    choices = [0 for _ in range(nb_arm)]
    nb_pull = [0 for _ in range(nb_arm)]
    arm_rewards = [0 for _ in range(nb_arm)]
    for i in range(1,nb_step+1):
        #print(f"Step {i}")
        R = np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        best_arm = 0
        satisfying_arm = None
        best_mean = float('-inf')
        for j in range(nb_arm):
            if nb_pull[j] > 0:
                mean = arm_rewards[j]/nb_pull[j]
                if mean-R/np.sqrt(nb_pull[j]) >= satisfaction_level:
                    satisfying_arm = j
                    break
                if mean > best_mean:
                    best_arm = j
                    best_mean = mean
            else:
                best_arm = j
                best_mean = float('inf')
        if satisfying_arm == None:
            if best_mean < satisfaction_level:
                chosen_arm = np.random.randint(0,nb_arm)
            else:
                chosen_arm = best_arm
            choices[chosen_arm] += 1
        else:
            chosen_arm = satisfying_arm
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations,choices


def experiment_ucb(bandits,satisfaction_level,nb_step,repetition=1):
    print("Experiment UCB for satisficing")
    rewards,regrets,expectations,choices = [],[],[],[]
    for bandit in progressbar(bandits):
        for _ in range(repetition):
            rew,reg,exp,cho = ucb_suffice(bandit, satisfaction_level, nb_step)
            rewards.append(rew)
            regrets.append(reg)
            expectations.append(exp)
            choices.append(cho)
    return np.mean(rewards,0),np.mean(regrets,0),np.mean(expectations,0),np.mean(choices,0)

def experiment_cb_ratio(bandits,satisfaction_level,nb_step,repetition=1):
    print("Experiment CB Ratio")
    rewards,regrets,expectations,choices = [],[],[],[]
    for bandit in progressbar(bandits):
        for _ in range(repetition):
            rew,reg,exp,cho = cb_ratio(bandit, satisfaction_level, nb_step)
            rewards.append(rew)
            regrets.append(reg)
            expectations.append(exp)
            choices.append(cho)
    return np.mean(rewards,0),np.mean(regrets,0),np.mean(expectations,0),np.mean(choices,0)

def experiment_mean_sat(bandits,satisfaction_level,nb_step,repetition=1):
    print("Experiment Optimistic_Mean_Satisficing")
    rewards,regrets,expectations,choices = [],[],[],[]
    for bandit in progressbar(bandits):
        for _ in range(repetition):
            rew,reg,exp,cho = optimistic_mean_sat(bandit, satisfaction_level, nb_step)
            rewards.append(rew)
            regrets.append(reg)
            expectations.append(exp)
            choices.append(cho)
    return np.mean(rewards,0),np.mean(regrets,0),np.mean(expectations,0),np.mean(choices,0)

def compute_bound_mean_sat(bandit,satisfaction_level):
    deltas = [satisfaction_level-m for m in bandit.means]
    R = 0
    for d in deltas:
        if d > 0:
            R += d + 2/d + 2*d/(np.abs(np.max(deltas))**2)
    return R

satisfaction_level = 0.7
nb_arm = 10
nb_step = 5000
nb_bandit = 1
repetition = 50

bandits = [GaussianBandit(nb_arm,threshold=satisfaction_level) for _ in range(nb_bandit)]

# rewards,regrets,expectations,choices = experiment_ucb(bandits, satisfaction_level, nb_step, repetition)
# plt.plot(np.cumsum(regrets))
# plt.title("Satisfaction regret for UCB-Satisficing")
# (plt.figure()
# plt.plot(expectations)
# plt.title("Expected reward for the chosen arm for UCB-Satisficing")
# plt.plot(np.ones_like(expectations)*satisfaction_level))

plt.figure()
rewards,regrets,expectations,choices = experiment_cb_ratio(bandits, satisfaction_level, nb_step, repetition)
plt.plot(np.cumsum(regrets))
plt.title("Satisfaction regret for CB-Ratio")
plt.figure()
plt.plot(expectations)
plt.title("Expected reward for the chosen arm for CB-Ratio")
plt.plot(np.ones_like(expectations)*satisfaction_level)


# plt.figure()
# rewards,regrets,expectations,choices = experiment_mean_sat(bandits, satisfaction_level, nb_step, repetition)
# plt.plot(np.cumsum(regrets))
# plt.title("Satisfaction regret for Optimistic-Mean-Satisficing")
# bound = compute_bound_mean_sat(bandits[0], satisfaction_level)
# plt.plot(np.ones_like(expectations)*bound)
# plt.yscale('linear')
# plt.figure()
# plt.plot(expectations)
# plt.title("Expected reward for the chosen arm for Optimistic_Mean_Satisficing")
# plt.plot(np.ones_like(expectations)*satisfaction_level)


plt.show()