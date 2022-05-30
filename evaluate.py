import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_state(t, last_action):
    data_chunk = data[t:t+window_size+1]
    return np.concatenate([sigmoid(np.diff(data_chunk)/data_chunk[1:] * 1000), last_action])

data = []
for i in range(1,15):
    data.append(np.load('data/data{}.npy'.format(i)))
data = np.concatenate(data)

model = keras.models.load_model('models/t-bot_112.h5')
model.compile()

actions = []
window_count = 0
data_samples = len(data) - 1
window_size =  60
bought = False
sold = False
buy_price = 0
balance = 0
balances = []
state = get_state(window_count, np.array([0,0,1]))
for timestep in tqdm(range(1, data_samples-window_size-2)):
    window_count += 1
    action_probs = model(np.array([state]), training=False)
    state = get_state(window_count+1, tf.cast(action_probs[0], tf.dtypes.int32))
    action = np.argmax(action_probs)
    
    # this code is used in case the bot issues alot of buy/sell signales -> sample a few of them.
    #if action == 1:
    #    if random.uniform(0, 1) > 0.5:
    #        action = 2
            
    #if action == 0:
    #    if random.uniform(0, 1) > 0.5:
    #        action = 2
            
    if action == 0 and bought:
        action = 2
    if action == 0 and not bought:
        bought = True
        sold = False
        buy_price = data[window_count]
        balance -= 0.1   
    if action == 1 and sold:
        action = 2
    if action == 1 and not sold and not bought:
        action = 2
    if action == 1 and not sold and bought:
        sold = True
        bought = False
        balance += (data[window_count]/buy_price - 1) * 100 - 0.1
        
    balances.append(balance)
    actions.append(action)

print('>> Final balance: ', balance)

buys = np.where(np.array(actions) == 0)
sells = np.where(np.array(actions) == 1)

print('>> Number of actions: ', np.unique(actions, return_counts=True))

fig, ax1 = plt.subplots(figsize=(16,5))
t = list(range(len(data)))

ax1.set_xlabel('time (minutes)')
ax1.set_ylabel('price', color='black')
ax1.plot(t, data, color='gray')
ax1.scatter(buys[0]+window_size, data[buys[0]+window_size], c='green', s=100)
ax1.scatter(sells[0]+window_size, data[sells[0]+window_size], c='red', s=100)
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('realized gain (%)', color=color)
ax2.plot(t, list(np.zeros(len(data) - len(balances)))+balances, color=color)
ax2.hlines(0, xmin=0, xmax=len(balances), colors='g', linestyles='-.')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig("chart.png", format="png", bbox_inches="tight")