import chess
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import pygame
import random
from pygame.locals import *

class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()

        # Action space: Index of legal moves
        self.action_space = spaces.Discrete(4672)  # Number of possible moves (max)
        # Observation space: Board state
        self.observation_space = spaces.Box(low=0, high=12, shape=(64,), dtype=np.int8)

        # Initialize pygame for rendering
        pygame.init()
        self.width, self.height = 480, 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chess Bot Training")

        # Font for rendering pieces
        pygame.font.init()
        self.font = pygame.font.SysFont('segoeuisymbol', 48)

        # Colors
        self.WHITE = pygame.Color(123, 99, 71)
        self.BLACK = pygame.Color(0, 0, 0)
        self.LIGHT_BROWN = pygame.Color(235, 235, 208)
        self.DARK_BROWN = pygame.Color(119, 149, 86)

        # Piece symbols
        self.piece_unicode = {
            'P': '♙',
            'N': '♘',
            'B': '♗',
            'R': '♖',
            'Q': '♕',
            'K': '♔',
            'p': '♟',
            'n': '♞',
            'b': '♝',
            'r': '♜',
            'q': '♛',
            'k': '♚',
        }

        self.tile_size = 60

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation."""
        self.board.reset()
        return self.get_board_state()

    def get_board_state(self):
        """Returns the current board state as a flat array."""
        board_state = np.zeros(64, dtype=np.int8)
        for square, piece in self.board.piece_map().items():
            board_state[square] = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
        return board_state

    def step(self, action):
        """Executes one time step within the environment."""
        legal_moves = list(self.board.legal_moves)
        action = action % len(legal_moves)
        # print(f"Legal moves: {[move.uci() for move in legal_moves]}")  # Debugging line
        # print(f"Action received: {action}")  # Debugging line
        if action < 0 or action >= len(legal_moves):
            # Invalid action
            reward = -10
            done = True
            return self.get_board_state(), reward, done, {}
        else:
            move = legal_moves[action]
            self.board.push(move)
            reward = 0
            done = self.board.is_game_over()
            if done:
                if self.board.result() == '1-0':
                    reward = 1  # Win
                elif self.board.result() == '0-1':
                    reward = -1  # Loss
                else:
                    reward = 0  # Draw
            return self.get_board_state(), reward, done, {}

    def render(self, mode='human'):
        """Renders the environment."""
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        self.screen.fill(self.WHITE)
        self.draw_board()
        self.draw_pieces()
        pygame.display.flip()

    def draw_board(self):
        """Draws the chess board."""
        colors = [self.LIGHT_BROWN, self.DARK_BROWN]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size))

    def draw_pieces(self):
        """Draws the pieces on the chess board using Unicode characters."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                piece_symbol = self.piece_unicode[piece.symbol()]
                text_surface = self.font.render(piece_symbol, True, self.BLACK if piece.color else self.WHITE)
                text_rect = text_surface.get_rect(center=((col + 0.5) * self.tile_size, (row + 0.5) * self.tile_size))
                self.screen.blit(text_surface, text_rect)

    def close(self):
        """Closes the environment."""
        pygame.quit()

env = ChessEnv()

# Wrap the environment to allow for render mode
from stable_baselines3.common.env_util import make_vec_env
vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)

# Training loop with rendering
TIMESTEPS = 1
for i in range(1, TIMESTEPS + 1):
    model.learn(total_timesteps=1000, reset_num_timesteps=False)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Step: {i}, Action: {action}, Reward: {reward}, Done: {done}")
        # pygame.time.wait(500)
        if done:
            pygame.time.wait(5000)
            break
        if env.board.is_game_over():
            env.reset()

env.close()