import matplotlib
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import math
import random
import pickle

class Agent():
	
	def __init__(self, ax):
		self.circle = Circle((0.50, 0.50), 0.025, color='blue')
		self.center = self.circle.center
		self.numEyes = 5
		self.viewDist = 0.2
		self.eyes = [[self.center[0], self.center[0] - (self.viewDist/2), self.center[1], self.center[1] + 0.866*self.viewDist],
				[self.center[0], self.center[0] - (self.viewDist * 0.259), self.center[1], self.center[1] + 0.966*self.viewDist],
				[self.center[0], self.center[0], self.center[1], self.center[1] + self.viewDist],
				[self.center[0], self.center[0] + (self.viewDist * 0.259), self.center[1], self.center[1] + 0.966*self.viewDist],
				[self.center[0], self.center[0] + (self.viewDist/2), self.center[1], self.center[1] + 0.866*self.viewDist]]
		self.axis = ax
		self.angle = math.pi / 2
		self.axis.add_artist(self.circle)
		self.eyesPlot = []
		for i in range(self.numEyes):
			temp1, = self.axis.plot(self.eyes[i][:2], self.eyes[i][2:], color='black')
			self.eyesPlot.append(temp1)

		self.epsilon = 0.1
		self.alpha = 0.2
		self.gamma = 0.9
		self.q = {}
		self.actions = [0, 1, 2, 3, 4] #forward, turn left little, turn right little, turn left more, turn right more
	
	def move(self, dist):
		distx = -math.sin(self.angle - math.pi / 2) * dist
		disty = math.cos(self.angle - math.pi / 2) * dist
		self.center = self.center[0] + distx, self.center[1] + disty
		self.circle.center = self.center

		for i in range(self.numEyes):
			self.eyes[i] = [self.center[0], self.eyes[i][1]+distx, self.center[1], self.eyes[i][3] + disty]
			self.eyesPlot[i].set_xdata(self.eyes[i][:2])
			self.eyesPlot[i].set_ydata(self.eyes[i][2:])

	def turn(self, angle):
		angle = angle * math.pi / 180
		for i in range(self.numEyes):
			newx = (self.eyes[i][1] - self.eyes[i][0]) * math.cos(angle) - (self.eyes[i][3] - self.eyes[i][2]) * math.sin(angle) + self.eyes[i][0]
			newy = (self.eyes[i][1] - self.eyes[i][0]) * math.sin(angle) + (self.eyes[i][3] - self.eyes[i][2]) * math.cos(angle) + self.eyes[i][2]
			self.eyes[i] = [self.eyes[i][0], newx, self.eyes[i][2], newy]
			self.eyesPlot[i].set_xdata(self.eyes[i][:2])
			self.eyesPlot[i].set_ydata(self.eyes[i][2:])
		self.angle = (self.angle + angle) % (2 * math.pi)

	def atEdge(self):
		for i in range(self.numEyes):
			if not (-self.viewDist + self.circle.radius * 2 < self.eyes[i][1] < self.axis.get_xlim()[1] + self.viewDist - self.circle.radius * 2 ) or \
			not (-self.viewDist + self.circle.radius * 2 < self.eyes[i][3] < self.axis.get_ylim()[1] + self.viewDist - self.circle.radius * 2):
				return True
		return False

	def nearEdge(self):
		for i in range(self.numEyes):
			if not (0 < self.eyes[i][1] < self.axis.get_xlim()[1]) or not (-0 < self.eyes[i][3] < self.axis.get_ylim()[1]):
				return True
		return False
	
	def getQ(self, state, action):
		return self.q.get((state, action), 0.0)

	def learnQ(self, state, action, reward, value):
		oldv = self.q.get((state, action), None)
		if oldv is None:
			self.q[(state, action)] = reward
		else:
			self.q[(state, action)] = oldv + self.alpha * (value - oldv)

	def chooseAction(self, state):
		if random.random() < self.epsilon:
			action = random.choice(self.actions)
		else:
			q = [self.getQ(state, a) for a in self.actions]
			maxQ = max(q)
			count = q.count(maxQ)
			if count > 1:
				best = [i for i in range(len(self.actions)) if q[i] == maxQ]
				i = random.choice(best)
			else:
				i = q.index(maxQ)
	
			action = self.actions[i]
		return action

	def learn(self, state1, action1, reward, state2):
		maxqnew = max([self.getQ(state2, a) for a in self.actions])
		self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)

	def saveQ(self):
		with open('model.pkl', 'wb') as f:
			pickle.dump(self.q, f)

	def loadQ(self):
		with open('model.pkl', 'rb') as f:
			self.q = pickle.load(f)

	
