import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas_datareader as data_reader
from tqdm import tqdm
import pandas as pd

seed = 42
gamma = 0.99  
epsilon = 1.0  
epsilon_min = 0.1 
epsilon_max = 1.0  
epsilon_interval = (epsilon_max - epsilon_min)  
batch_size = 64 
max_steps_per_episode = 10000
num_actions = 3
window_size = 60
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
optimizer = keras.optimizers.Adam(learning_rate=0.00025)
window_count = 0
epsilon_random_states = 256
epsilon_greedy_states = 256
max_memory_length = data_samples / 2
update_after_actions = 8
update_target_network = 128
loss_function = keras.losses.Huber()
minimum_gain = 1 # percent
willing_to_wait = 120 # days

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_state(t, last_action):
    data_chunk = data[t:t+window_size+1]
    return np.concatenate([sigmoid(np.diff(data_chunk)/data_chunk[1:] * 1000), last_action])

def get_reward(action, last_state, t, inventary, total_gains):
    now = t+window_size+1
    current_price = data[now]
    future = data[now: now+willing_to_wait]
    last_action = np.argmax(last_state[-3:])
    # buy
    if action == 0:
        if last_action == 0: return 0, inventary, total_gains
        diff = (np.max(future)/current_price - 1)* 100
        reward = max(0,  diff - minimum_gain)
        inventary.append(current_price)
    # sell
    elif action == 1:
        if last_action == 1 or not inventary: return 0, inventary, total_gains
        diff = (np.mean(future)/np.mean(inventary) - 1)* 100
        reward = max(0, diff - minimum_gain)
        inventary = []
        total_gains += diff
    # hold
    else:
        if last_action == 1 or not inventary: return 0, inventary, total_gains
        diff = (np.mean(future)/np.mean(inventary) - 1)* 100
        reward = max(0, diff - minimum_gain)
    return reward, inventary, total_gains

def create_q_model_():
    # slow...
    visible = layers.Input(shape=(window_size,1))
    hidden1 = layers.LSTM(32)(visible)
    output = layers.Dense(num_actions, activation='linear')(hidden1)
    return keras.Model(inputs=visible, outputs=output)

def create_q_model():
    inputs = layers.Input(shape=(window_size + num_actions,))
    layer1 = layers.Dense(64, activation="relu")(inputs)
    layer2 = layers.Dense(64, activation="relu")(layer1)
    layer3 = layers.Dense(32, activation="relu")(layer2)
    action = layers.Dense(num_actions, activation="softmax")(layer3)
    return keras.Model(inputs=inputs, outputs=action)

model = create_q_model()
model_target = create_q_model()
data = []
# tested with 14 days
for i in range(1,15):
    data.append(np.load('data/data{}.npy'.format(i)))
data = np.concatenate(data)
data_samples = len(data) - 1

while True: 
    state = get_state(window_count, np.array([0,0,1]))
    episode_reward = 0
    inventary = []
    total_gains = 0
    window_count = 0
    n_actions = 0
    for timestep in tqdm(range(1, data_samples-window_size-2)):
        window_count += 1
        if window_count < epsilon_random_states or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
            action_probs = np.array([[0,0,0]])
            action_probs[0][action] = 1
        else:
            state_tensor = tf.convert_to_tensor(state)
            action_probs = model(np.array([state_tensor]), training=False)
            action = tf.argmax(action_probs[0]).numpy()
        epsilon -= epsilon_interval / epsilon_greedy_states
        epsilon = max(epsilon, epsilon_min)
        reward, inventary, total_gains = get_reward(action, state, window_count, inventary, total_gains)
        state_next = get_state(window_count+1, tf.cast(action_probs[0], tf.dtypes.int32))
        episode_reward += reward
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(0)
        rewards_history.append(reward)
        state = state_next
        if window_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([done_history[i] for i in indices])
            future_rewards = model_target.predict(state_next_sample)
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if window_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())

        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 128:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    c, u = np.unique(action_history, return_counts=True)
    template = "running reward: {:.2f} at episode {}, actions: {}/{}. Total gains: {}"
    print(template.format(running_reward, episode_count, c, u, total_gains))

    model.save("models/t-bot_{}.h5".format(episode_count))