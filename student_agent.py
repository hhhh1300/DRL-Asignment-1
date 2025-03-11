# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random

# 全域變數：只在首次呼叫 get_action(obs) 時初始化
Q_table = None           
previous_obs = None      # 上一次的 obs
passenger_in_taxi = 0    # 上一次推斷的「是否已載客」，0 or 1

def update_passenger_in_taxi(prev_obs, current_obs, prev_in_taxi):
    """
    依賴連續兩個 obs，推斷是否已載客。
    - obs[14] => passenger_look (0: 乘客不在同格/相鄰, 1: 在同格或相鄰)
    - 若 passenger_look=1 且計程車位置改變 => 可能是已經載起乘客。
    - 若 passenger_look=0 => 表示乘客不在這 => 必定沒載客。
    """
    if prev_obs is None:
        # 第一次呼叫，還沒上一個 obs，可以視為沒載客
        return 0

    prev_passenger_look = prev_obs[14]
    curr_passenger_look = current_obs[14]
    prev_taxi_row, prev_taxi_col = prev_obs[0], prev_obs[1]
    curr_taxi_row, curr_taxi_col = current_obs[0], current_obs[1]

    # case 1: 如果現在 passenger_look=0，表示乘客不在此 => 一定沒載客
    if curr_passenger_look == 0:
        return 0

    # case 2: passenger_look=1
    if prev_in_taxi == 1:
        # 已經在載客狀態，就維持
        return 1
    else:
        # 先前還沒載客，但現在 passenger_look=1
        # 如果上一個 obs 也 passenger_look=1，且計程車位置改變 => 代表乘客「跟著」動 => 推斷已載客
        if prev_passenger_look == 1 and (curr_taxi_row != prev_taxi_row or curr_taxi_col != prev_taxi_col):
            return 1
        else:
            return 0

def get_action(obs):
    """
    從 obs (shape=16) + 全域記憶 (previous_obs, passenger_in_taxi)
    來推斷當前是否已載客。再用 (obs, passenger_in_taxi) 查 Q-table 得到動作。
    如果沒對應 key，就隨機動作 fallback。
    """
    global Q_table
    global previous_obs
    global passenger_in_taxi

    # 第一次呼叫時，載入訓練好的 Q_table
    if Q_table is None:
        with open("q_table.pkl", "rb") as f:
            Q_table = pickle.load(f)

    # 1) 根據前後 obs，更新 passenger_in_taxi
    passenger_in_taxi = update_passenger_in_taxi(previous_obs, obs, passenger_in_taxi)

    # 2) 將 obs (array) 轉成 tuple，並加上 passenger_in_taxi，形成 state_key
    obs_key = tuple(map(int, obs))  # obs 16 維整數化
    state_key = obs_key + (passenger_in_taxi,)

    # 3) 查表，若無 key 則隨機動作
    try:
        if np.random.uniform(0, 1) < 0.1:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            q_values = Q_table[state_key]
            action = int(np.argmax(q_values))
    except KeyError:
        action = random.choice([0, 1, 2, 3, 4, 5])

    # 4) 更新 previous_obs
    previous_obs = obs

    return action
