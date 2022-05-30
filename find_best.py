import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
import random

# trading rules:
# you are allowed to buy only if you didn't buy before
# you are allowed to sell only if bought already
# each trade has a fee of 0.1%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_state(t, last_action):
    data_chunk = data[t:t+window_size+1]
    return np.concatenate([sigmoid(np.diff(data_chunk)/data_chunk[1:] * 1000), last_action])

data = np.load('data/data1.npy')

storage = []
data_samples = len(data) - 1
window_size =  60

for i in range(1, 123):
    try:
        model = keras.models.load_model('models/t-bot_{}.h5'.format(i))
        model.compile()
        actions = []
        window_count = 0
        bought = False
        sold = False
        buy_price = 0
        balance = 0
        balances = []
        state = get_state(window_count, np.array([0,0,1]))
        for timestep in range(1, data_samples-window_size-2):
            window_count += 1
            action_probs = model(np.array([state]), training=False)
            state = get_state(window_count+1, tf.cast(action_probs[0], tf.dtypes.int32))
            action = np.argmax(action_probs)
            if action == 0 and bought:
                action = 2
            if action == 1 and not sold and not bought:
                action = 2
            if action == 0 and not bought:
                bought = True
                sold = False
                buy_price = data[window_count]
                balance -= 0.1
            if action == 1 and sold:
                action = 2
            if action == 1 and not sold and bought:
                sold = True
                bought = False
                balance += (data[window_count]/buy_price - 1) * 100 - 0.1
            balances.append(balance)
            actions.append(action)
        print(i, balance)
        buys = np.where(np.array(actions) == 0)
        sells = np.where(np.array(actions) == 1)
        storage.append(balances)
    except:
        print(i)

print('>> Best model: ', np.argmax(np.array(storage)[:,-1]))
print('>> Best gain: ', np.max(np.array(storage)[:,-1]))