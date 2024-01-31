import gym
import numpy as np
from ple import PLE
from ple.games.contsubmarinegame import ContSubmarineGame  # Adjust the import according to your file structure
import pygame
import os

def process_state(state):
    # Convert the dictionary state to a numpy array if needed
    return np.array(list(state.values()))

class ContSubmarineGameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, display_screen=True):
        self.game = ContSubmarineGame()
        self.ple = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.display.set_mode((1, 1))
        self.ple.init()
        self.action_set = self.ple.getActionSet()
        self.action_space = gym.spaces.Discrete(len(self.action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.game.getScreenDims(), dtype=np.uint8)
        self.viewer = None

    def step(self, action):
        reward = self.ple.act(self.action_set[action])
        state = self.ple.getGameState()
        done = self.ple.game_over()
        return state, reward, done, {}

    def reset(self):
        self.ple.reset_game()
        return self.ple.getGameState()

    def render(self, mode='human', close=False):
        if close:
            pygame.quit()
            return
        elif mode == 'rgb_array':
            return self.ple.getScreenRGB()
        elif mode == 'human':
            # Implement human mode rendering if needed
            pass

    def close(self):
        pygame.quit()
