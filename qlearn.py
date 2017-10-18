import matplotlib
from matplotlib.patches import Circle
import Agent
import matplotlib.pyplot as plt
import random
import math
import argparse

def create_world():
	dots = []
	for i in range(50):
		x, y = genRandPt(dots)
		while x == y == -1:
			x, y = genRandPt(dots)
		dot = create_dot(x, y)
		ax.add_artist(dot)
		dots.append(dot)
	return dots

def genRandPt(dots):
	# 0.015  is radius of dot
	x = random.uniform(0.015, ax.get_xlim()[1] - 0.015)
	y = random.uniform(0.015, ax.get_ylim()[1] - 0.015)
	for dot in dots:
		dx = abs(dot.center[0] - x)
		dy = abs(dot.center[1] - y)
		if dx < dot.radius * 2 and dy < dot.radius * 2:
			return -1, -1
	return x, y

def create_dot(x, y):
	if random.random() < 0.6:
		color = 'green'
	else:
		color = 'red'
	dot = Circle((x, y), 0.015, color=color)
	return dot

def dotDetected():
	dist = [99] * agent.numEyes
	detected = [0] * agent.numEyes
	wall = agent.nearEdge()
	for i,eye in enumerate(agent.eyes):
		tempDist = distToWall(eye)
		if tempDist < 99:
			detected[i] = 3
			dist[i] = tempDist				
		for dot in dots:
			if eyeSeeDot(dot, eye):
				tempDist = pt2ptDist(dot.center[0], dot.center[1], agent.circle.center[0], agent.circle.center[1], dot.radius, agent.circle.radius)
				if tempDist < dist[i]:
					dist[i] = tempDist
					if dot.get_facecolor()[0] == 0.0:	# green
						detected[i] = 1
					elif dot.get_facecolor()[0] == 1.0:	# red
						detected[i] = 2
	return detected

def distToWall(eye):
	if eye[1] < 0:
		angle = math.asin(min(abs(eye[1] - eye[0]) / agent.viewDist, 1))
		if angle == 0:
			dist = agent.viewDist - abs(eye[1])
		else:
			dist = agent.viewDist - abs(eye[1]) / math.sin(angle)
	elif eye[1] > ax.get_xlim()[1]:
		angle = math.asin(min((eye[1] - eye[0]) / agent.viewDist, 1))
		if angle == 0:
			dist = agent.viewDist - (eye[1] - ax.get_xlim()[1])
		else:
			dist = agent.viewDist - (eye[1] - ax.get_xlim()[1]) / math.sin(angle)
	elif eye[3] < 0:
		angle = math.acos(min(abs(eye[3] - eye[2]) / agent.viewDist, 1))
		if angle == math.pi / 2:
			dist = agent.viewDist - abs(eye[3])
		else:
			dist = agent.viewDist - abs(eye[3]) / math.cos(angle)
	elif eye[3] > ax.get_ylim()[1]:
		angle = math.acos(min((eye[3] - eye[2]) / agent.viewDist, 1))
		if angle == math.pi / 2:
			dist = agent.viewDist - (eye[3] - ax.get_ylim()[1])
		else:
			dist = agent.viewDist - (eye[3] - ax.get_ylim()[1]) / math.cos(angle)
	else:
		dist = 99
	return dist
	
def pt2ptDist(x1, y1, x2, y2, radius1, radius2):
	dist = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
	dist = dist - radius1 - radius2
	return dist

def eyeSeeDot(dot, eye):
	A = dot.center[0] - eye[0]
	B = dot.center[1] - eye[2]
	C = eye[1] - eye[0]
	D = eye[3] - eye[2]
	
	dotprod = A * C + B * D
	lenSq = C * C + D * D
	param = dotprod / lenSq
	
	if param < 0:
		xx = eye[0]
		yy = eye[2]
	elif param > 1:
		xx = eye[1]
		yy = eye[3]
	else:
		xx = eye[0] + param * C
		yy = eye[2] + param * D

	dx = dot.center[0] - xx
	dy = dot.center[1] - yy
	
	dist = math.sqrt(dx * dx + dy * dy)
	return dist < dot.radius
	

def dotAbsorbed():
	absorbed = []
	for i, dot in enumerate(dots):
		if (agent.center[0] - agent.circle.radius < dot.center[0] < agent.center[0] + agent.circle.radius) and (agent.center[1] - agent.circle.radius < dot.center[1] < agent.center[1] + agent.circle.radius):
			absorbed.append(i)
	return absorbed

def smoothMove(delay):
	agent.move(0.0125)
	fig.canvas.draw()
	plt.pause(delay / 2)
	agent.move(0.0125)

def smoothTurn(angle, delay):
	agent.turn(float(angle) / 2)
	fig.canvas.draw()
	plt.pause(delay / 2)
	agent.turn(float(angle) / 2)

def train(delay, iters):
	detected = dotDetected()
	lastState = None
	lastAction = None
	reward = 0.0
	score = 0
	age = 0
	dotsCollected = [0]*5000
	greenCollected = [0]*5000
	prevAtEdge = False
		
	while age < iters:
		state = tuple(detected)		# + (agent.nearEdge(),)
		action = agent.chooseAction(state)
		reward = 0

		if action == 0:
			if not agent.atEdge():
				agent.move(0.025)
				if agent.nearEdge():
					reward -= 0.1
				else:			
					reward += 0.5
					if prevAtEdge:
						reward += 0.25
				prevAtEdge = False
			else:
				prevAtEdge = True
				reward -= 1.0
		elif action == 1:
			agent.turn(15)
			if not agent.atEdge():
				agent.move(0.025)
		elif action == 2:
			agent.turn(-15)
			if not agent.atEdge():
				agent.move(0.025)
		elif action == 3:
			agent.turn(30)
			if not agent.atEdge():
				agent.move(0.01)
		elif action == 4:
			agent.turn(-30)
			if not agent.atEdge():
				agent.move(0.01)
	
		absorbed = dotAbsorbed()
		dotsCollected[age%5000] = 0
		greenCollected[age%5000] = 0
		for dot in absorbed:
			if (dots[dot].get_facecolor()[0] == 1.0):
				reward += -6.0
			else:
				reward += 5.0
				greenCollected[age%5000] += 1
			x, y = genRandPt(dots)
			while x == y == -1:
				x, y = genRandPt(dots)

			dots[dot].center = (x, y)
			dotAges[dot] = 0
			dotsCollected[age%5000] += 1
		
		detected = dotDetected()
	
		for i in range(agent.numEyes):
			if detected[i] == 0:
				agent.eyesPlot[i].set_color('black')
			elif detected[i] == 1:
				agent.eyesPlot[i].set_color('green')
			elif detected[i] == 2:
				agent.eyesPlot[i].set_color('red')
			elif detected[i] == 3:
				agent.eyesPlot[i].set_color('yellow')

		fig.canvas.draw()
		plt.pause(delay)
	
		if lastState is not None:
			agent.learn(lastState, lastAction, reward, state)
		lastState = state
		lastAction = action
	
		for i in range(len(dotAges)):
			if dotAges[i] > 2500 and age % 100 == 0 and random.random() < 0.05:
				x, y = genRandPt(dots)
				while x == y == -1:
					x, y = genRandPt(dots)

				dots[i].center = (x, y)
				dotAges[i] = 0
			else:
				dotAges[i] += 1

		score += reward
		age += 1
		dcSum = 0
		gcSum = 0
		for i in range(5000):
			dcSum += dotsCollected[i]
			gcSum += greenCollected[i]
		if dcSum > 0:
			fuzzScore = gcSum * 1.0 / dcSum
		else:
			fuzzScore = 0.0
		plt.title("age=%d  ratio=%.3f  score=%d" % (age, fuzzScore, score))

	agent.saveQ()


def play(delay):
	detected = dotDetected()
	age = 0
	dotsCollected = 0
	greenCollected = 0
	
	agent.loadQ()
	agent.epsilon = 0.0

	while True:
		state = tuple(detected)		# + (agent.nearEdge(),)
		action = agent.chooseAction(state)
		if action == 0:
			if not agent.atEdge():
				#agent.move(0.025)
				smoothMove(delay)
		elif action == 1:
			#agent.turn(15)
			smoothTurn(15, delay)
			if not agent.atEdge():
				smoothMove(delay)
				prevAtEdge = False
			else:
				prevAtEdge = True
		elif action == 2:
			#agent.turn(-15)
			smoothTurn(-15, delay)
			if not agent.atEdge():
				smoothMove(delay)
				prevAtEdge = False
			else:
				prevAtEdge = True
		elif action == 3:
			#agent.turn(30)
			smoothTurn(30, delay)
			if not agent.atEdge():
				agent.move(0.01)
				prevAtEdge = False
			else:
				prevAtEdge = True
		elif action == 4:
			#agent.turn(-30)
			smoothTurn(-30, delay)
			if not agent.atEdge():
				agent.move(0.01)
				prevAtEdge = False
			else:
				prevAtEdge = True
	
		absorbed = dotAbsorbed()

		for dot in absorbed:
			if (dots[dot].get_facecolor()[0] == 0.0):
				greenCollected += 1
			x, y = genRandPt(dots)
			while x == y == -1:
				x, y = genRandPt(dots)

			dots[dot].center = (x, y)
			dotAges[dot] = 0
			dotsCollected += 1
		
		detected = dotDetected()
	
		for i in range(agent.numEyes):
			if detected[i] == 0:
				agent.eyesPlot[i].set_color('black')
			elif detected[i] == 1:
				agent.eyesPlot[i].set_color('green')
			elif detected[i] == 2:
				agent.eyesPlot[i].set_color('red')
			elif detected[i] == 3:
				agent.eyesPlot[i].set_color('yellow')

		fig.canvas.draw()
		plt.pause(delay / 2)
	
		for i in range(len(dotAges)):
			if dotAges[i] > 2500 and age % 100 == 0 and random.random() < 0.05:
				x, y = genRandPt(dots)
				while x == y == -1:
					x, y = genRandPt(dots)

				dots[i].center = (x, y)
				dotAges[i] = 0
			else:
				dotAges[i] += 1

		age += 1
		if dotsCollected > 0:
			fuzzScore = float(greenCollected) / dotsCollected
		else:
			fuzzScore = 0.0
		plt.title("ratio=%.3f" % (fuzzScore))

if __name__ == "__main__":
	print("Parsing Args")
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", help="Mode is either 'train' to train a model or 'play' to load a trained model", required=False, default="train")
	parser.add_argument("-s", "--speed", help="Control how fast the animation is between 1 (slowest) and 5 (fastest)", required=False, type=int, default=3)
	parser.add_argument("-i", "--iterations", help="Number of iterations to train before saving a model", required=False, type=int, default=50000)
	args = vars(parser.parse_args())

	fig, ax = plt.subplots(1, 1)
	print("Generating world")
	dots = create_world()
	dotAges = [0]*len(dots)
	agent = Agent.Agent(ax)

	ax.set_aspect('equal')
	plt.ylim([0, 1])
	plt.xlim([0, 1])
	#plt.ion()
	plt.show(block=False)
	ax.set_yticklabels([])
	ax.set_xticklabels([])

	mode = args['mode']
	speed = args['speed']
	iters = args['iterations']

	timeDelays = [0.5, 0.2, 0.1, 0.05, 0.01]
	delay = timeDelays[speed - 1]

	if mode == "train":
		try:
			train(delay, iters)
		except KeyboardInterrupt:
			print("User cancelled training. No model saved.")
	elif mode == "play":
		try:
			play(delay)
		except KeyboardInterrupt:
			print("User ended session.")

	




