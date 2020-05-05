import time

import numpy as np
import tensorflow as tf

from agentBreakOut import Agent
from deepQNetwork import buildq_network
from gameWrapper import GameWrapper
from hyperparameters import TOTAL_FRAMES, FRAMES_BETWEEN_EVAL, MAX_EPISODE_LENGTH, CLIP_REWARD, UPDATE_FREQ, \
    EVAL_LENGTH, WRITE_TENSORBOARD, BATCH_SIZE, DISCOUNT_FACTOR, MIN_REPLAY_BUFFER_SIZE, SAVE_PATH, PRIORITY_SCALE, \
    MAX_NOOP_STEPS, LEARNING_RATE, INPUT_SHAPE, MEM_SIZE
from loadClasses import writer, game_wrapper, agent,frame_number,rewards,loss_list
from replayBuffer import ReplayBuffer

ENV_NAME = 'BreakoutDeterministic-v4'

# Create environment
game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

# Create agent
MAIN_DQN = buildq_network(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = buildq_network(game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

print('Loading model...')
agent.load('save-01603987')
print('Loaded')

terminal = True
eval_rewards = []
evaluate_frame_number = 0

for frame in range(EVAL_LENGTH):
    if terminal:
        game_wrapper.reset(evaluation=True)
        life_lost = True
        episode_reward_sum = 0
        terminal = False

    # Breakout require a "fire" action (action #1) to start the
    # game each time a life is lost.
    # Otherwise, the agent would sit around doing nothing.
    action = 1 if life_lost else agent.get_action(0, game_wrapper.state, evaluation=True)

    # Step action
    _, reward, terminal, life_lost = game_wrapper.step(action, render_mode='human')
    evaluate_frame_number += 1
    episode_reward_sum += reward

    # On game-over
    if terminal:
        print(f'Game over, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
        eval_rewards.append(episode_reward_sum)

print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)