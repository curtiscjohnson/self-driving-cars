import gym
import numpy as np
from gym import spaces
from simulation import Simulator
import random


class CustomDuckieTownSim(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        camera_settings,
        map_parameters,
        car_parameters,
        action_angles: list = [-30, 0, 30],
        display=False,
    ):
        super().__init__()

        self.camera_settings = camera_settings
        self.map_parameters = map_parameters
        self.car_parameters = car_parameters
        self.display = display

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_angles = action_angles
        N_DISCRETE_ACTIONS = len(action_angles)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input (channel-first; channel-last also works):
        N_CHANNELS = 3
        (HEIGHT, WIDTH) = self.camera_settings["resolution"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8
        )
        # ! I'm pretty sure observation space is supposed to be the features the agent has access to -- the preprocessed image.

    def preprocess_img(self, raw_img):
        # some feature engineering to separate out red/white/yellow was done in that paper
        # maybe also do some horizon cropping? maybe not important for sim training
        # also maybe stacking a short sequence of images too? - https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecframestack
        # !SB3 CNNPolicy normalizes images by default.
        return raw_img

    def step(self, action):
        raw_img, reward, self.done = self.sim.step(
            steer=self.action_angles[action], speed=1.0, display=self.display
        )
        self.info = {}

        observation = self.preprocess_img(raw_img)
        return observation, reward, self.done, self.info

    def reset(self):
        """Reset gets called right after init typically.
        This is actually where most of the setup comes in.

        Returns:
            _type_: _description_
        """

        self.done = False

        self.sim = Simulator(cameraSettings=self.camera_settings)

        startLocations = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [1, 0],
                [1, 1],
                [1, 4],
                [1, 7],
                [2, 2],
                [2, 3],
                [1, 4],
                [2, 5],
                [2, 6],
                [2, 0],
                [5, 1],
                [3, 2],
                [2, 4],
                [5, 5],
                [5, 6],
                [2, 7],
                [3, 0],
                [7, 1],
                [4, 2],
                [7, 3],
                [5, 4],
                [6, 5],
                [3, 7],
                [4, 0],
                [5, 2],
                [7, 4],
                [7, 5],
                [4, 7],
                [5, 0],
                [7, 2],
                [5, 7],
                [6, 0],
            ]
        )
        startLoc = random.randint(0, 38)

        self.sim.start(
            mapSeed="real",
            mapParameters=self.map_parameters,
            carParameters=self.car_parameters,
            startPoint=(
                int(startLocations[startLoc, 0]),
                int(startLocations[startLoc, 1]),
                0,
                0,
            ),
        )

        where, facing = self.sim.RealSense.parent.ackermann.pose()
        initial_img = self.sim.RealSense.camera.getImage(where, facing)

        observation = self.preprocess_img(initial_img)
        return observation  # reward, done, info can't be included
