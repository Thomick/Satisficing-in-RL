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

class BernoulliBandit(Bandit):
    def __init__(self, nb_arm, range = [0, 1], threshold = 0.7, means=None):
        super().__init__(nb_arm,range=range,threshold=threshold,means=means)
    
    def pull(self,arm):
        if np.random.rand() < self.means[arm]:
            return 1.
        else:
            return 0.

def ucb(bandit,satisfaction_level,nb_step,parameter=1):
    confidence_multiplier = parameter
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
        R = confidence_multiplier*np.sqrt(2*np.log(1+i*(np.log(i)**2)))
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

def algo1(bandit,satisfaction_level,nb_step,parameter=1):
    confidence_multiplier = parameter
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
        R = confidence_multiplier*np.sqrt(2*np.log(1+i*(np.log(i)**2)))
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



def algo3(bandit,satisfaction_level,nb_step,parameter=1):
    confidence_multiplier = parameter
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
        R = confidence_multiplier*np.sqrt(2*np.log(1+i*(np.log(i)**2)))
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

def algo3xucb(bandit,satisfaction_level,nb_step,parameter=1):
    confidence_multiplier = parameter
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
        R = confidence_multiplier*np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        confidence = np.ones(nb_arm)*R / np.sqrt(nb_pull)
        emp_avg = arm_rewards/nb_pull
        ucb = emp_avg + confidence
        lcb = emp_avg - confidence
        ratio = (ucb - np.maximum(lcb,satisfaction_level))/confidence

        max_ucb_arm = np.argmax(ucb)
        m = satisfaction_level
        mi = -1
        for i in range(ucb.shape[0]): # Find arm with highest ucb among the one with average reward > satisfaction level
            if ucb[i]>= m and emp_avg[i] >= satisfaction_level:
                m = ucb[i]
                mi = i
        if mi != -1:
            chosen_arm = mi
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


def algo3xavg(bandit,satisfaction_level,nb_step,parameter=1):
    confidence_multiplier = parameter
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
        R = confidence_multiplier*np.sqrt(2*np.log(1+i*(np.log(i)**2)))
        confidence = np.ones(nb_arm)*R / np.sqrt(nb_pull)
        emp_avg = arm_rewards/nb_pull
        ucb = emp_avg + confidence
        lcb = emp_avg - confidence
        ratio = (ucb - np.maximum(lcb,satisfaction_level))/confidence

        max_avg_arm = np.argmax(emp_avg)
        if emp_avg[max_avg_arm]>= satisfaction_level:
            chosen_arm = max_avg_arm
        else:
            max_ucb_arm = np.argmax(ucb)
            chosen_arm = max_ucb_arm

        # Update rewards and metrics
        reward = bandit.pull(chosen_arm)
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations

def bernoulli_thompson(bandit,satisfaction_level,nb_step,parameter=1):
    nb_arm = bandit.nb_arm
    rewards = [] # Empirical reward at each step
    regrets = [] # Satisfying regret at each step
    expectations = [] # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm) # Number of pull of each arm
    arm_rewards = np.zeros(nb_arm) # Empirical total reward of each arm
    alpha = np.ones(nb_arm)
    beta = np.ones(nb_arm)

    for i in range(nb_step):
        theta = np.random.beta(alpha, beta)
        chosen_arm = np.argmax(theta)

        # Get reward
        reward = bandit.pull(chosen_arm)
        # Update distriutions
        alpha[chosen_arm] += reward
        beta[chosen_arm] += 1 - reward
        # Update metrics
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations

def epsilon_greedy(bandit,satisfaction_level,nb_step,parameter=1):
    epsilon = parameter
    nb_arm = bandit.nb_arm
    rewards = [] # Empirical reward at each step
    regrets = [] # Satisfying regret at each step
    expectations = [] # Expected reward at each step (for the played policy)
    nb_pull = np.zeros(nb_arm) # Number of pull of each arm
    arm_rewards = np.zeros(nb_arm) # Empirical total reward of each arm

    for i in range(nb_step):
        if np.random.uniform(0,1) > epsilon:
            chosen_arm = np.argmax(arm_rewards/np.maximum(nb_pull,1))
        else:
            chosen_arm = np.random.randint(0, nb_arm)

        # Get reward
        reward = bandit.pull(chosen_arm)
        # Update metrics
        nb_pull[chosen_arm] += 1
        arm_rewards[chosen_arm] += reward
        rewards.append(reward)
        regrets.append(max(0,satisfaction_level-bandit.means[chosen_arm]))
        expectations.append(bandit.means[chosen_arm])
    return rewards,regrets,expectations

def experiment(algo,bandits,satisfaction_level,nb_step,nb_repetition=1,parameter=1):
    print(f"Experiment ({nb_repetition} runs)")
    rewards,regrets,expectations= [],[],[]
    for bandit in bandits:  # In case we want to average over multiple bandit instances
        for _ in progressbar(range(nb_repetition)):  # Repetition of the experiment on the bandit instance
            rew,reg,exp = algo(bandit, satisfaction_level, nb_step,parameter=parameter)
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


