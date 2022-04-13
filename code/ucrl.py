import numpy as np
import random as rd
import copy as cp

# Vanilla UCRL2 based on Jacksh 2010
# Implementation from Hippolyte Bourel
# Contains:
#	nS the number of state
#	nA the number of action
#	t the global time
#	delta the given parameter delta in [0, 1]
#	observations the set of the 3 lists of the observed states, actions and rewards each ordered by time
#	vk the state-action count for the current episode k
#	Nk the state-action count prior to episode k
#	policy the current estimated policy
#	r( and p)_distances are the confidence bounds on the estimates.
class UCRL2:
	def __init__(self,nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))

	def name(self):
		return "UCRL2"

	# Auxiliary function to compute N the current state-action count.
	def computeN(self):
		N = np.zeros((self.nS, self.nA))
		for t in range(len(self.observations[1])):
			N[self.observations[0][t], self.observations[1][t]] += 1
		self.Nk = N
	
	# Auxiliary function to compute R the current accumulated reward.
	def computeR(self):
		R = np.zeros((self.nS, self.nA))
		for t in range(len(self.observations[1])):
			R[self.observations[0][t], self.observations[1][t]] += self.observations[2][t]
		return R
	
	# Auxiliary function to compute P the current transitions count.
	def computeP(self):
		P = np.zeros((self.nS, self.nA, self.nS))
		for t in range(len(self.observations[1])):
			P[self.observations[0][t], self.observations[1][t], self.observations[0][t+1]] += 1
		return P
	
	#Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
	def distances(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
												/ (2 * max([1, self.Nk[s, a]])))
				self.p_distances[s, a] = np.sqrt((14 * self.nS * np.log(2 * self.nA * self.t / self.delta))
												/ (max([1, self.Nk[s, a]])))
		
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(p_estimate[s, a])
			max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])# Error?
				l += 1
		return max_p
	
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01):
		u0 = np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		while True:
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = r_estimate[s, a] + self.r_distances[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.vk = np.zeros((self.nS, self.nA))
		self.computeN()
		Rk = self.computeR()
		Pk = self.computeP()
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = Pk[s, a, next_s] / div
		self.distances()
		self.EVI(r_estimate, p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[state, action] += 1
		self.t += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)