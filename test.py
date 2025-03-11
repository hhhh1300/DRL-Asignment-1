import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle
from collections import defaultdict

# 假設您的 SimpleTaxiEnv 定義在 simple_taxi_env.py
from simple_custom_taxi_env import SimpleTaxiEnv

# ---------------------------
# 1. 建立 DQN 的網路結構
# ---------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 這裡是簡單的兩層全連接 (MLP)，可自行調整層數與神經元數
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)


# ---------------------------
# 2. 設定 Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ---------------------------
# 3. 超參數設定
# ---------------------------
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0       # 初始 epsilon-greedy
EPS_END = 0.01        # 最低 epsilon
EPS_DECAY = 10000     # 多少 step 後衰減到最低
TARGET_UPDATE_FREQ = 1000  # 每隔多少 step 更新 target network
REPLAY_BUFFER_SIZE = 100000
MAX_EPISODES = 500    # 可自行調整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 4. 訓練函式
# ---------------------------
def update_passenger_in_taxi(prev_obs, current_obs, prev_passenger_in_taxi):
    """
    根據上一個 obs 與當前 obs，推斷是否已載客 (0 or 1)，純粹依賴 obs。
    - prev_obs[14] / current_obs[14] => passenger_look (0 or 1)
      代表乘客是否跟計程車在同格/相鄰
    - 利用「計程車坐標是否變動 + passenger_look」來判斷是否成功載客。
    """
    if prev_obs is None:
        # 第一個 step 還無法比較，預設沒載客
        return 0
    
    # passenger_look 當前
    curr_passenger_look = current_obs[14]
    # passenger_look 前一刻
    prev_passenger_look = prev_obs[14]
    
    # 取出計程車位置
    taxi_row, taxi_col = current_obs[0], current_obs[1]
    prev_taxi_row, prev_taxi_col = prev_obs[0], prev_obs[1]
    
    # 若現在 passenger_look = 0 => 代表乘客不在這 (或不相鄰)，肯定沒載客
    if curr_passenger_look == 0:
        return 0
    
    # 若現在 passenger_look = 1
    # 如果先前就已經在載客狀態，則持續保持
    if prev_passenger_in_taxi == 1:
        return 1
    
    # 否則 (prev_passenger_in_taxi=0)，檢查前後 obs 是否都 passenger_look=1，
    # 且計程車位置有改變 => 表示乘客隨車移動 => 推斷已載客
    if prev_passenger_look == 1 and (taxi_row != prev_taxi_row or taxi_col != prev_taxi_col):
        return 1
    
    # 其他情況 => 尚未載客
    return 0

def make_state_key(obs, passenger_in_taxi):
    """
    將 obs + passenger_in_taxi 組合成適合儲存在字典中的 key。
    注意: 這裡只示範取 obs 全部 16 維再加 1 維 passenger_in_taxi => 17 維。
    您也可以只取部分欄位，只要「訓練」與「測試」時一致即可。
    """
    # 轉成 tuple
    # obs 原本可能是 numpy array，需要轉成 tuple or list
    obs_tuple = tuple(int(x) for x in obs)
    # 最後加上 passenger_in_taxi
    return obs_tuple + (passenger_in_taxi,)

def train_q_table():
    """
    使用 Q-Learning 進行訓練，並將 Q-table 存成 q_table.pkl。
    """

    # 1) 建立環境
    env_config = {
        "fuel_limit": 5000
    }
    env = SimpleTaxiEnv(**env_config)

    # 2) 建立 Q-table (dict)，預設每個動作的 Q 值 = 0
    #    也可使用 collections.defaultdict(lambda: np.zeros(6))
    Q_table = defaultdict(lambda: np.zeros(6))

    # 3) 超參數
    num_episodes = 500            # 可自行調整
    alpha = 0.1                   # 學習率
    gamma = 0.99                  # 折扣因子
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 10000   # 多少 step 後衰減到最低
    epsilon = epsilon_start

    # 為了 epsilon 衰減
    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        # 初始化 passenger_in_taxi = 0
        passenger_in_taxi = 0
        previous_obs = None
        
        # 建立初始 state key
        state_key = make_state_key(obs, passenger_in_taxi)
        
        done = False
        episode_reward = 0
        steps_in_episode = 0

        while not done:
            steps_in_episode += 1
            total_steps += 1

            # Epsilon-greedy 選動作
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                q_values = Q_table[state_key]
                action = int(np.argmax(q_values))

            # 與環境互動
            next_obs, reward, done, _ = env.step(action)

            # 依照 obs 變化來推斷下一個 passenger_in_taxi
            next_passenger_in_taxi = update_passenger_in_taxi(previous_obs, obs, passenger_in_taxi)

            # 組合出 next state key
            next_state_key = make_state_key(next_obs, next_passenger_in_taxi)

            # Q-Learning 更新
            best_next_q = np.max(Q_table[next_state_key])  # 下個狀態最好的 Q
            Q_table[state_key][action] += alpha * (reward + gamma * best_next_q - Q_table[state_key][action])

            # 狀態推進
            passenger_in_taxi = next_passenger_in_taxi
            previous_obs = obs
            obs = next_obs
            state_key = next_state_key
            episode_reward += reward

            # Epsilon 衰減 (線性)
            if epsilon > epsilon_end:
                epsilon -= (epsilon_start - epsilon_end) / epsilon_decay_steps
                epsilon = max(epsilon, epsilon_end)
        
        print(f"Episode {ep+1}/{num_episodes}, Steps: {steps_in_episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    # 訓練結束後，將 Q_table 存成 pkl 檔 (dict 格式)
    Q_table = dict(Q_table)  # defaultdict 無法直接 dump，先轉成普通 dict
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)
    print("Training finished, Q-table saved to q_table.pkl!")


if __name__ == "__main__":
    train_q_table()