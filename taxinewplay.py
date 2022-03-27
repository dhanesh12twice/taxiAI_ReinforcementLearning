import gym
import numpy as np
import random

env = gym.make("Taxi-v3")

qtable = np.loadtxt("taxi_qtable.txt",dtype=float)
max_steps = 99 # per episode

state = env.reset()
done = False
rewards = 0

for s in range(max_steps):
    print("TRAINED AGENT")
    print("Step {}".format(s+1))
    
    action = np.argmax(qtable[state,:])
    new_state, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    
    print("score: {}".format(rewards))
    state = new_state
    
    if done == True:
        break

env.close()
