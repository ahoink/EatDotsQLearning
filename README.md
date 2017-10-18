# EatDotsQLearning
A simple implementation of q-learning in python

**About**

A blue circle learns to navigate a field to eat green dots and avoid red dots. The learning process involves q-learning, an off-policy reinforcement learning algorithm based on states and actions. 

The blue circle is the q-learning agent. The agent has five "eyes" angle 15 degrees from each other and thus can see in five directions. The eyes can detect three things (or nothing): green dots, red dots, and the edges of the field. With five eyes that can see four different things (including nothing), there are total of 20 possible states the agent can be in.

The agent can perform five different actions: move straight forward, turn a little to the left and move forward, turn a little to the right and move forward, turn more to the left and move forward a little, or turn more to the right and move forward a little.

The agent is reward for eating green dots, moving straight forward, and moving away from the edges. The agent is penalized for eating red dots, moving too close to the edge, and hitting the edge.

Eventually, the agent learns which actions are "good" based on its current state and chooses actions it thinks is best based on both immediate and future rewards it will get from performing said action.

All of the above attributes can be modified which can affect learning rate and performance. 

**Requirements**
- Python 3 (has not been tested for 2.7.x but should be easily adaptable
- matplotlib

**Running**

python qlearn.py [options]
