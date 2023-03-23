import gym
import numpy as np
from gym import spaces
from simulation import Simulator
import random
from img_utils import preprocess_image as image_process
from collections import deque
import cv2

import PID_Code.lightning_mcqueen as lm


class CustomDuckieTownSim(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        camera_settings,
        map_parameters,
        car_parameters,
        NUM_BLOBS = 5,
        action_angles: list = [-30, 0, 30],
        SPEED= 1.0,
        randomizeCameraParamsOnReset=False,
        display=False,
    ):
        super().__init__()

        self.camera_settings = camera_settings
        self.map_parameters = map_parameters
        self.car_parameters = car_parameters
        self.display = display
        self.SPEED = SPEED
        self.NUM_BLOBS = NUM_BLOBS

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_angles = action_angles
        N_DISCRETE_ACTIONS = len(action_angles)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.observation_space = spaces.Box(
            low=0, high=max(self.camera_settings['resolution']), shape=(NUM_BLOBS, 2), dtype=np.uint8)

        # ! I'm pretty sure observation space is supposed to be the features the agent has access to -- the preprocessed image.
        self.randomizeCameraParamOnReset = randomizeCameraParamsOnReset
        self.initial_obs = np.ones((NUM_BLOBS, 2))*-1

    def step(self, action):
        raw_img, reward, self.done = self.sim.step(
            steer=self.action_angles[action], speed=self.SPEED, display=self.display
        )
        self.info = {}

        observation = lm.get_yellow_centers(raw_img)
        final_obs = observation[-self.NUM_BLOBS:]
        # print(observation)
        # print(final_obs)
        # lm.draw_centers(raw_img, observation, size=7)

        if observation == "None":
            final_obs = self.initial_obs
        else:
            num_observations = len(final_obs)
            if num_observations < self.NUM_BLOBS:
                for i in range(self.NUM_BLOBS - num_observations):
                    final_obs.append((-1,-1))

        # print(np.asarray(final_obs))

        # cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
        # cv2.imshow('raw', raw_img)
        # cv2.waitKey(0)
        return final_obs, reward, self.done, self.info

    def reset(self):
        """Reset gets called right after init typically.
        This is actually where most of the setup comes in.

        Returns:
            _type_: _description_
        """

        self.done = False
        self.num_steps_taken = 0

        if self.randomizeCameraParamOnReset:
            self.camera_settings["angle"]["pitch"] = np.random.uniform(-10, 0)
            self.camera_settings["angle"]["roll"] = np.random.uniform(-5,5)

            # print(self.camera_settings["angle"]["pitch"])
            # print(self.camera_settings["angle"]["roll"])

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

        # where, facing = self.sim.RealSense.parent.ackermann.pose()
        # initial_img = self.sim.RealSense.camera.getImage(where, facing)
        # print(f"initial_img.shape: {initial_img.shape}")
        # observation = self.preprocess_img(initial_img)
        # print(f"observation.shape: {observation.shape}")
        # self.observation_buffer.append(observation)

        return self.initial_obs
