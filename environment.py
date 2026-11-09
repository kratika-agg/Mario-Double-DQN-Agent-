
import gym
import numpy as np
import torch

from torchvision import transforms as T
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        height, width = self.observation_space.shape[:2]

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(1, height, width),
            dtype=np.uint8
        )

        self.transform = T.Grayscale()

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)

        observation = self.transform(observation)

        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(1, *self.shape),
            dtype=np.float32
        )

        self.transform = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])

    def observation(self, observation):
        observation = self.transform(observation)
        return observation


def create_mario_env(render_mode="rgb"):
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode=render_mode,
        apply_api_compatibility=True
    )

    env = JoypadSpace(
        env,
        [["right"], ["right", "A"]]
    )

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    return env
