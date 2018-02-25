import gym
import numpy as np
from os import getcwd
from DQN import load_dqn
from random import randint
from keras.models import load_model
    
# Select Environment
env = gym.make('MountainCar-v0')

# Initialize Environment Settings
number_episodes = 10
max_timesteps = 1000
past_timeframes_view = 4

# Determine State and Action Spaces
state_dim = (past_timeframes_view, ) + env.observation_space.shape
action_dim = env.action_space.n

# Initialize Reinforcement Learning Hyperparameters
exploration_coef = 0.05
gamma = 0.95

# Initialize Replay Memory
replay_memory = []

# Initialize Session Settings
session = 0
dqn_network_name = "MountainCarNetwork"
new_dqn_network = True
learning = True
played_episodes = 0

# Initialize Deep Q Network
if new_dqn_network:
    # Build DQN From Scratch
    dqn = load_dqn(state_dim, action_dim)
else: 
    # Load Previous Model
    model_path = getcwd() + "/" + dqn_network_name + ".h5"
    dqn = load_model(model_path)

# Run Episodes
for e in range(number_episodes):

    # Load Initial State
    original_observation = env.reset()
    observation = np.expand_dims(original_observation, axis=0)
    state = np.expand_dims(np.vstack([observation] * past_timeframes_view), axis=0)

    for t in range(max_timesteps):
        # Render Game
        print("Episode {} - Iteration {}".format(played_episodes + e + 1, t+1))
        env.render()

        # Select Approach - Exploitation vs Exploration
        if np.random.rand() > exploration_coef or not learning:
            # Exploit - Choose best action
            result = dqn.predict(state)
            action = np.argmax(result)
        else:
            # Explore - Choose random action
            action = env.action_space.sample()

        # Execute Action
        new_observation, reward, done, info = env.step(action)

        # Build New State (Stack of the last few observations)
        new_observation = np.expand_dims(new_observation, axis=0)
        new_state = np.roll(state, -1, axis=1)
        new_state[0][-1] = new_observation

        # Reinforcement Learning
        if learning:
            # Store Previous Experience in Replay Memory
            replay_memory.append([state, action, new_state, reward])

            # Recall a Random Past Experience
            memory_index = randint(0, len(replay_memory) - 1)
            recall_state, recall_action, recall_new_state, recall_reward = replay_memory[memory_index]
            
            # Predict Q values for Possible Actions
            targets = np.zeros((action_dim))
            targets = dqn.predict(recall_state)

            # Adjust Q Value of Action Taken according to Reward Received
            if done:
                targets[0][recall_action] = reward
                dqn.fit(recall_state, targets)
                break
            else:
                targets[0][recall_action] = reward + gamma * np.max(dqn.predict(recall_new_state))
                dqn.fit(recall_state, targets)

        # Update State
        state = new_state

# Save Model
trained_model_path = getcwd() + "/" + dqn_network_name + "-S{}".format(session + 1) + ".h5"
dqn.save(trained_model_path)