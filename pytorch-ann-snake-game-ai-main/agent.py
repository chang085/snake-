import json
import os
import random
from collections import deque

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from game import Game
#from game_no_ui import Game


# ======================================================
# MẠNG THẦN KINH NHÂN TẠO (ANN)
# ======================================================
class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ======================================================
# BỘ NHỚ KINH NGHIỆM
# ======================================================
class ReplayMemory:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, k):
        experiences = random.sample(self.memory, k=k)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones


# ======================================================
# SIÊU THAM SỐ
# ======================================================
number_episodes = 100000
maximum_number_steps_per_episode = 20000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.001
epsilon_decay_value = 0.995
learning_rate = 0.001
minibatch_size = 128
gamma = 0.95
replay_buffer_size = int(1e5)
tau = 1e-2
state_size = 16
action_size = 4
scores_on_100_episodes = deque(maxlen=100)
folder = "model"


# ======================================================
# AGENT
# ======================================================
class Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_network = ANN(state_size, action_size).to(self.device)
        self.target_network = ANN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
        self.record = -1
        self.epsilon = -1

    def get_state(self, game):
        head_x, head_y = game.snake.x[0], game.snake.y[0]

        point_left = [head_x - game.BLOCK_WIDTH, head_y]
        point_right = [head_x + game.BLOCK_WIDTH, head_y]
        point_up = [head_x, head_y - game.BLOCK_WIDTH]
        point_down = [head_x, head_y + game.BLOCK_WIDTH]
        point_left_up = [head_x - game.BLOCK_WIDTH, head_y - game.BLOCK_WIDTH]
        point_left_down = [head_x - game.BLOCK_WIDTH, head_y + game.BLOCK_WIDTH]
        point_right_up = [head_x + game.BLOCK_WIDTH, head_y - game.BLOCK_WIDTH]
        point_right_down = [head_x + game.BLOCK_WIDTH, head_y + game.BLOCK_WIDTH]

        state = [
            game.is_danger(point_left),
            game.is_danger(point_right),
            game.is_danger(point_up),
            game.is_danger(point_down),
            game.is_danger(point_left_up),
            game.is_danger(point_left_down),
            game.is_danger(point_right_up),
            game.is_danger(point_right_down),

            game.snake.direction == "left",
            game.snake.direction == "right",
            game.snake.direction == "up",
            game.snake.direction == "down",

            game.apple.x < head_x,   # food left
            game.apple.x > head_x,   # food right
            game.apple.y < head_y,   # food up
            game.apple.y > head_y,   # ✅ FIXED: food down
        ]
        return np.array(state, dtype=int)

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > minibatch_size:
            experiences = self.memory.sample(k=minibatch_size)
            self.learn(experiences)

    def get_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()
        if random.random() > epsilon:
            move = torch.argmax(action_values).item()
        else:
            move = random.randint(0, 3)
        return move

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_q_targets = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * next_q_targets * (1 - dones)
        q_expected = self.local_network(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_network, self.target_network)

    def soft_update(self, local_network, target_network):
        for local_params, target_params in zip(local_network.parameters(), target_network.parameters()):
            target_params.data.copy_(tau * local_params + (1.0 - tau) * target_params)

    def save_model(self, file_name="model.pth"):
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, file_name)
        torch.save({
            'model_state_dict': self.local_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)

    def load(self, file_name="model.pth"):
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            self.local_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Model Loaded")
            self.retrieve_data()

    def save_data(self, record, epsilon):
        data_path = os.path.join(folder, "data.json")
        with open(data_path, "w") as file:
            json.dump({'record': record, 'epsilon': epsilon}, file, indent=4)

    def retrieve_data(self):
        data_path = os.path.join(folder, "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data = json.load(file)
                self.record = data.get("record", -1)
                self.epsilon = data.get("epsilon", -1)


# ======================================================
# HUẤN LUYỆN + BIỂU ĐỒ ĐỘNG
# ======================================================
def train():
    game = Game()
    agent = Agent(state_size, action_size)
    agent.load()
    max_score = 0
    epsilon = agent.epsilon if agent.epsilon != -1 else epsilon_starting_value

    print(f"🔹 Epsilon khởi tạo: {epsilon:.3f}")

    # --- chuẩn bị biểu đồ ---
    plt.ion()
    fig, ax = plt.subplots()
    scores, avgs = [], []
    line1, = ax.plot(scores, label='Curr Score')
    line2, = ax.plot(avgs, label='Avg Score (100 ep)')
    ax.legend()
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    try:
        for episode in range(number_episodes):
            game.reset()
            score = 0
            for t in range(maximum_number_steps_per_episode):
                state_old = agent.get_state(game)
                action = agent.get_action(state_old, epsilon)
                move = [0, 0, 0, 0]
                move[action] = 1
                reward, done, score = game.run(move)
                state_new = agent.get_state(game)
                agent.step(state_old, action, reward, state_new, done)
                if done:
                    break
            max_score = max(max_score, score)
            scores_on_100_episodes.append(score)
            epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

            # --- cập nhật biểu đồ ---
            scores.append(score)
            avgs.append(np.mean(scores_on_100_episodes))
            line1.set_xdata(np.arange(len(scores)))
            line1.set_ydata(scores)
            line2.set_xdata(np.arange(len(avgs)))
            line2.set_ydata(avgs)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

            if episode % 50 == 0:
                print(f"Ep {episode}\tScore {score}\tMax {max_score}\tAvg {np.mean(scores_on_100_episodes):.2f}")

            agent.save_model()
            agent.save_data(max_score, epsilon)

    except KeyboardInterrupt:
        print("\n🛑 Huấn luyện dừng bằng Ctrl+C. Đang lưu model...")
        agent.save_model()
        agent.save_data(max_score, epsilon)
        print("✅ Model đã lưu an toàn.")
    finally:
        plt.ioff()
        plt.show()


# ======================================================
# KIỂM TRA MODEL
# ======================================================
def test(n_episodes=200):
    # Dùng phiên bản có pygame UI
    game = Game()
    agent = Agent(state_size, action_size)
    agent.load()
    epsilon = 0.0  # test không ngẫu nhiên

    print("\n🚀 Bắt đầu TEST MODE (epsilon = 0) với giao diện\n")

    # --- setup biểu đồ ---
    plt.ion()
    fig, ax = plt.subplots()
    scores, avgs = [], []
    line1, = ax.plot(scores, label='Curr Score', color='tab:blue')
    line2, = ax.plot(avgs, label='Avg Score (50 ep)', color='tab:orange')
    ax.legend()
    plt.title("Testing Progress (With UI)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    scores_window = deque(maxlen=50)

    try:
        for ep in range(1, n_episodes + 1):
            game.reset()
            score = 0
            while True:
                # lấy state hiện tại
                state = agent.get_state(game)
                action = agent.get_action(state, epsilon)

                move = [0, 0, 0, 0]
                move[action] = 1
                reward, done, score = game.run(move)

                if done:
                    break

            scores_window.append(score)
            scores.append(score)
            avgs.append(np.mean(scores_window))

            # --- cập nhật biểu đồ ---
            line1.set_xdata(np.arange(len(scores)))
            line1.set_ydata(scores)
            line2.set_xdata(np.arange(len(avgs)))
            line2.set_ydata(avgs)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

            print(f"Episode {ep}/{n_episodes} → Score: {score}\tAvg(50): {np.mean(scores_window):.2f}")

        print("\n✅ Test hoàn tất.")
    except KeyboardInterrupt:
        print("\n🛑 Dừng test thủ công (Ctrl + C).")
    finally:
        plt.ioff()
        plt.show()
        import pygame
        pygame.quit()



# ======================================================
if __name__ == "__main__":
    #train()
    test()
