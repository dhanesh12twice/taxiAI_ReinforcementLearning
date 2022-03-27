import numpy as np
import gym
import random

def main():

    # Taxi environment
    env = gym.make('Taxi-v3')

    # Initialization q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # Variables
    num_episodes = 1000
    max_steps = 99 # per episode

    # Actual Training
    for episode in range(num_episodes):

        # Resetting the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # Exploration-Exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit
                action = np.argmax(qtable[state,:])

            # action and reward
            new_state, reward, done, info = env.step(action)

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Updating state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

    print("Training completed over {} episodes".format(num_episodes))
    np.savetxt("taxi_qtable.txt", qtable)
   

if __name__ == "__main__":
    main()


