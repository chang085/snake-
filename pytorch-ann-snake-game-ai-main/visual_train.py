import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from agent import Agent, state_size, action_size, scores_on_100_episodes
from game_no_ui import Game

# --- Shared variables ---
current_score = 0
avg_score = 0
stop_flag = False


def training_thread():
    global current_score, avg_score, stop_flag
    game = Game()
    agent = Agent(state_size=state_size, action_size=action_size)
    agent.load()

    epsilon = 1.0 if agent.epsilon == -1 else agent.epsilon
    max_score = 0

    print("Starting training loop...")

    for episode in range(1, 5000):  # Giới hạn ví dụ
        if stop_flag:
            break

        game.reset()
        score = 0

        for _ in range(10000):
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, epsilon)
            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.run(move)
            state_new = agent.get_state(game)
            agent.step(state_old, action, reward, state_new, done)

            if done:
                break

        scores_on_100_episodes.append(score)
        current_score = score
        avg_score = np.mean(scores_on_100_episodes)

        epsilon = max(0.001, 0.99 * epsilon)
        max_score = max(max_score, score)

        if episode % 50 == 0:
            print(f"Episode {episode} | Curr Score {score} | Avg {avg_score:.2f}")

    stop_flag = True
    print("Training finished.")


# --- Visualization Thread ---
scores = []
averages = []

def update_plot(frame):
    scores.append(current_score)
    averages.append(avg_score)
    plt.cla()
    plt.plot(scores, label='Curr Score', color='orange')
    plt.plot(averages, label='Average (100 episodes)', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Training Progress (Snake AI)")
    plt.tight_layout()


if __name__ == "__main__":
    # Start training in background thread
    t = threading.Thread(target=training_thread)
    t.start()

    # Create live chart
    fig = plt.figure()
    ani = FuncAnimation(fig, update_plot, interval=1000)  # update mỗi giây
    plt.show()

    # Khi tắt biểu đồ -> dừng huấn luyện
    stop_flag = True
    t.join()
