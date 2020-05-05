# Create environment
import tensorflow as tf
from deepQNetwork    import buildq_network
from replayBuffer    import ReplayBuffer
from agentBreakOut   import Agent
from gameWrapper     import GameWrapper
from hyperparameters import ENV_NAME, MAX_NOOP_STEPS, TENSORBOARD_DIR, LOAD_FROM, LOAD_REPLAY_BUFFER,LEARNING_RATE,INPUT_SHAPE,MEM_SIZE,USE_PER,BATCH_SIZE

game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

# TensorBoard writer
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# Build main and target networks
MAIN_DQN = buildq_network(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = buildq_network(game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, use_per=USE_PER)

# Training and evaluation
if LOAD_FROM is None:
    frame_number = 0
    rewards = []
    loss_list = []
else:
    print('Loading from', LOAD_FROM)
    meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

    # Apply information loaded from meta
    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']

    print('Loaded')