import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from collections import deque

# 假設 SimpleTaxiEnv 定義在 simple_taxi_env.py
from simple_custom_taxi_env import SimpleTaxiEnv

# ----------------------------
# 1. 建立 DQN 網路結構
# ----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)


# ----------------------------
# 2. 建立 Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
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


# ----------------------------
# 3. 依據前後 obs 推斷 passenger_in_taxi
# ----------------------------
def update_passenger_in_taxi(prev_obs, current_obs, prev_in_taxi):
    """
    只能依賴 obs 本身 (尤其 obs[14] => passenger_look),
    以及計程車位置是否改變，來推斷是否已載客 (1) 或尚未載客 (0)。
    """
    if prev_obs is None:
        # 第一個 step 還沒有上一步的 obs
        return 0

    prev_passenger_look = prev_obs[14]
    curr_passenger_look = current_obs[14]
    prev_taxi_row, prev_taxi_col = prev_obs[0], prev_obs[1]
    curr_taxi_row, curr_taxi_col = current_obs[0], current_obs[1]

    # 若現在 passenger_look=0 => 一定沒載客
    if curr_passenger_look == 0:
        return 0

    # 若 passenger_look=1
    if prev_in_taxi == 1:
        # 先前已載客 => 維持
        return 1
    else:
        # 先前尚未載客，但現在 passenger_look=1
        # 若前後 obs 都是 passenger_look=1，且計程車座標改變 => 代表乘客「跟著」移動 => 推斷已載客
        if prev_passenger_look == 1 and (curr_taxi_row != prev_taxi_row or curr_taxi_col != prev_taxi_col):
            return 1
        else:
            return 0


# ----------------------------
# 4. DQN 訓練函式
# ----------------------------
def train_dqn():
    # === 建立環境 ===
    env_config = {
        "fuel_limit": 5000
    }
    env = SimpleTaxiEnv(**env_config)

    # === 超參數設定 ===
    state_dim = 17  # obs (16 維) + passenger_in_taxi (1 維)
    action_dim = 6  # 環境動作空間 (0~5)
    max_episodes = 50000
    max_steps_per_episode = 1000  # 給個安全上限，避免極端情況卡住

    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    buffer_capacity = 100000
    target_update_freq = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay_steps = 2000000

    # === 建立 DQN 與目標網路 ===
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    epsilon = eps_start
    total_steps = 0

    for ep in range(max_episodes):
        obs, _ = env.reset()
        # 我們需要維護上一個 obs (prev_obs) 與 passenger_in_taxi
        prev_obs = None
        passenger_in_taxi = 0

        # 建立初始 state
        state = np.concatenate([obs, [passenger_in_taxi]]).astype(np.float32)

        episode_reward = 0.0
        done = False
        t = 0
        
        for t in range(max_steps_per_episode):
            total_steps += 1

            # 1) epsilon-greedy 動作選擇
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = int(q_values.argmax(dim=1).item())

            # 2) 與環境互動
            next_obs, reward, done, _ = env.step(action)
            if done:
                if reward < 40:
                    done = False

            # 3) 用 prev_obs + obs => 推斷下一刻 passenger_in_taxi
            next_passenger_in_taxi = update_passenger_in_taxi(prev_obs, obs, passenger_in_taxi)

            # 建立 next_state
            next_state = np.concatenate([next_obs, [next_passenger_in_taxi]]).astype(np.float32)

            # 4) 儲存至 Replay Buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # 5) 狀態往後推
            prev_obs = obs
            obs = next_obs
            passenger_in_taxi = next_passenger_in_taxi
            state = next_state
            episode_reward += reward

            # 6) 取樣小批量做 DQN 更新
            if len(replay_buffer) >= batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

                batch_states_tensor = torch.FloatTensor(batch_states)
                batch_actions_tensor = torch.LongTensor(batch_actions).unsqueeze(1)
                batch_rewards_tensor = torch.FloatTensor(batch_rewards)
                batch_next_states_tensor = torch.FloatTensor(batch_next_states)
                batch_dones_tensor = torch.FloatTensor(batch_dones)

                # 計算 Q(s,a)
                q_values = policy_net(batch_states_tensor)  # (B, action_dim)
                current_q = q_values.gather(1, batch_actions_tensor).squeeze(1)

                # 計算 target Q(s',a')
                with torch.no_grad():
                    next_q_values = target_net(batch_next_states_tensor)
                    max_next_q = next_q_values.max(dim=1)[0]
                    target_q = batch_rewards_tensor + (1 - batch_dones_tensor) * gamma * max_next_q

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 定期同步 target_net
                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # 7) epsilon 衰減
            if epsilon > eps_end:
                epsilon -= (eps_start - eps_end) / eps_decay_steps
                epsilon = max(epsilon, eps_end)

            if done:
                break

        print(f"Episode {ep+1}/{max_episodes}, Steps: {t+1}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
        with open("trained_dqn_taxi.pkl", "wb") as f:
            pickle.dump(policy_net.state_dict(), f)

    # ---------------------------
    # 6. 簡單測試
    # ---------------------------
    test_episodes = 5
    print("=== Start Testing ===")
    for i in range(test_episodes):
        obs, _ = env.reset()
        prev_obs = None
        passenger_in_taxi = 0
        state = np.concatenate([obs, [passenger_in_taxi]]).astype(np.float32)

        done = False
        total_r = 0.0
        steps_in_episode = 0

        while not done:
            steps_in_episode += 1

            # 利用 policy_net 做動作選擇 (greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = int(q_values.argmax(dim=1).item())

            next_obs, reward, done, _ = env.step(action)

            # 更新 passenger_in_taxi
            next_passenger_in_taxi = update_passenger_in_taxi(prev_obs, obs, passenger_in_taxi)
            next_state = np.concatenate([next_obs, [next_passenger_in_taxi]]).astype(np.float32)

            prev_obs = obs
            obs = next_obs
            passenger_in_taxi = next_passenger_in_taxi
            state = next_state
            total_r += reward

            if steps_in_episode > 5000:  # 給個安全上限，避免意外卡住
                break

        print(f"[Test Episode {i+1}] Steps: {steps_in_episode}, TotalReward: {total_r:.2f}")


if __name__ == "__main__":
    train_dqn()
