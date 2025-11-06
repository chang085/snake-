import random
import numpy as np
import pygame
from base import Base


class Snake(Base):
    def __init__(self, parent_screen, length=1):
        super().__init__()
        self.length = length
        self.parent_screen = parent_screen
        self.block = pygame.image.load("resources/block.jpg")
        self.x = [self.BLOCK_WIDTH] * self.length
        self.y = [self.BLOCK_WIDTH] * self.length
        self.direction = "right"

    def draw(self):
        self.parent_screen.fill((0, 0, 0))
        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i]))

    def increase(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)

    def move_left(self):
        if self.direction != "right":
            self.direction = 'left'

    def move_right(self):
        if self.direction != "left":
            self.direction = 'right'

    def move_up(self):
        if self.direction != "down":
            self.direction = 'up'

    def move_down(self):
        if self.direction != "up":
            self.direction = 'down'

    def move(self):
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        if self.direction == 'right':
            self.x[0] += self.BLOCK_WIDTH
        elif self.direction == 'left':
            self.x[0] -= self.BLOCK_WIDTH
        elif self.direction == 'up':
            self.y[0] -= self.BLOCK_WIDTH
        elif self.direction == 'down':
            self.y[0] += self.BLOCK_WIDTH

        self.draw()


class Apple(Base):
    def __init__(self, parent_screen):
        super().__init__()
        self.parent_screen = parent_screen
        self.apple_img = pygame.image.load("resources/apple.jpg")
        self.x = self.BLOCK_WIDTH * 4
        self.y = self.BLOCK_WIDTH * 5

    def draw(self):
        self.parent_screen.blit(self.apple_img, (self.x, self.y))

    def move(self, snake):
        while True:
            x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            clean = True
            for i in range(0, snake.length):
                if x == snake.x[i] and y == snake.y[i]:
                    clean = False
                    break
            if clean:
                self.x = x
                self.y = y
                return


class Game(Base):
    def __init__(self, speed=300):
        super().__init__()
        pygame.init()
        pygame.display.set_caption("Snake Game - DQN Agent (UI Mode)")
        self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        self.snake = Snake(self.surface, length=1)
        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False
        self.reward = 0
        self.speed = speed  # tốc độ hiển thị (ms giữa các bước)
        self.clock = pygame.time.Clock()

    def play(self):
        self.snake.move()
        self.apple.draw()
        self.display_score()
        pygame.display.flip()

        # reward mặc định
        self.reward = -0.1

        # ăn táo
        if self.snake.x[0] == self.apple.x and self.snake.y[0] == self.apple.y:
            self.score += 1
            self.snake.increase()
            self.apple.move(self.snake)
            self.reward = 10

        # va chạm -> game over
        if self.is_collision():
            self.game_over = True
            self.reward = -100

    def is_collision(self):
        head_x = self.snake.x[0]
        head_y = self.snake.y[0]

        # chạm thân
        for i in range(1, self.snake.length):
            if head_x == self.snake.x[i] and head_y == self.snake.y[i]:
                return True

        # chạm tường
        if (head_x >= self.SCREEN_SIZE or head_y >= self.SCREEN_SIZE
                or head_x < 0 or head_y < 0):
            return True

        return False

    def is_danger(self, point):
        px, py = point
        for i in range(1, self.snake.length):
            if px == self.snake.x[i] and py == self.snake.y[i]:
                return True
        if (px >= self.SCREEN_SIZE or py >= self.SCREEN_SIZE
                or px < 0 or py < 0):
            return True
        return False

    def display_score(self):
        font = pygame.font.SysFont('arial', 20)
        text = font.render(f"Score: {self.score}", True, (200, 200, 200))
        self.surface.blit(text, (480, 10))

    def reset(self):
        self.snake = Snake(self.surface)
        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False

    def get_next_direction(self, move):
        # ["right", "down", "left", "up"]
        if np.array_equal(move, [1, 0, 0, 0]):
            return "right"
        elif np.array_equal(move, [0, 1, 0, 0]):
            return "down"
        elif np.array_equal(move, [0, 0, 1, 0]):
            return "left"
        elif np.array_equal(move, [0, 0, 0, 1]):
            return "up"
        return "right"

    def run(self, move):
        # Nhận hành động từ agent
        direction = self.get_next_direction(move)
        if direction == "left":
            self.snake.move_left()
        elif direction == "right":
            self.snake.move_right()
        elif direction == "down":
            self.snake.move_down()
        elif direction == "up":
            self.snake.move_up()

        # Gọi luôn play() mỗi vòng (như game_no_ui)
        self.play()

        # Kiểm tra thoát game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # điều chỉnh tốc độ (tuỳ chọn)
        if self.speed > 0:
            self.clock.tick(self.speed)

        return self.reward, self.game_over, self.score
