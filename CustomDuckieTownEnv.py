import gym
import numpy as np
from gym import spaces
from simulation import Simulator
import random
import cv2
from img_utils import preprocess_image as image_process
from collections import deque
from copy import deepcopy


class CustomDuckieTownSim(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        camera_settings,
        map_parameters,
        car_parameters,
        action_angles: list = [-30, 0, 30],
        addYellowNoise=False,
        blackAndWhite=False,
        use3imgBuffer=False,
        randomizeCameraParamsOnReset=False,
        yellow_features_only=False,
        column_mask=False,
        display=False,
    ):
        super().__init__()

        self.camera_settings = camera_settings
        self.map_parameters = map_parameters
        self.car_parameters = car_parameters
        self.display = display
        self.yellow_features_only = yellow_features_only
        self.column_mask = column_mask

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_angles = action_angles
        N_DISCRETE_ACTIONS = len(action_angles)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input (channel-first; channel-last also works):
        N_CHANNELS = 3

        # (HEIGHT, WIDTH) = self.camera_settings["resolution"]
        (WIDTH, HEIGHT) = self.camera_settings["resolution"]

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8
        )
        # ! I'm pretty sure observation space is supposed to be the features the agent has access to -- the preprocessed image.
        self.addYellowNoise = addYellowNoise
        self.blackAndWhite = blackAndWhite
        self.use3imgBuffer = use3imgBuffer
        self.steering_angle = 0
        self.randomizeCameraParamOnReset = randomizeCameraParamsOnReset
        if use3imgBuffer:
            self.initial_obs = [np.zeros((HEIGHT, WIDTH))] * 3
            self.observation_buffer = deque(self.initial_obs, maxlen=3)
        else:
            self.initial_obs = np.zeros((HEIGHT, WIDTH, N_CHANNELS))

    def preprocess_img(self, raw_img):
        # some feature engineering to separate out red/white/yellow was done in that paper
        # maybe also do some horizon cropping? maybe not important for sim training
        # also maybe stacking a short sequence of images too? - https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecframestack
        # !SB3 CNNPolicy normalizes images by default.

        processed_img = image_process(
            raw_img,
            removeBottomStrip=True,
            blackAndWhite=self.blackAndWhite,
            addYellowNoise=self.addYellowNoise,
            use3imgBuffer=self.use3imgBuffer,
            yellow_features_only=self.yellow_features_only,
            column_mask=self.column_mask,
            camera_resolution=tuple(self.camera_settings['resolution'])
        )

        return processed_img.astype(np.uint8)

    def step(self, action):
        self.steering_angle+=self.action_angles[action]/3
        self.steering_angle = np.clip(self.steering_angle, -30, 30)
        
        raw_img, reward, self.done = self.sim.step(
            steer=self.action_angles[action], speed=0.8, display=self.display
        )
        self.info = {}
        #! sim.step() returns raw image as height x width x channels.

        observation = self.preprocess_img(raw_img.copy())

        if self.use3imgBuffer:
            self.observation_buffer.append(observation)

        cv2.namedWindow('observation', cv2.WINDOW_NORMAL)
        cv2.imshow('observation', observation)
        cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
        cv2.imshow('raw', raw_img)
        cv2.waitKey(0)


        if self.use3imgBuffer:
            return np.dstack(self.observation_buffer), reward, self.done, self.info
        else:
            return observation, reward, self.done, self.info

    def reset(self):
        """
        Reset gets called right after init typically.
        This is actually where most of the setup comes in.

        Returns:
            _type_: _description_
        """

        
        self.done = False
        self.steering_angle = 0
        self.num_steps_taken = 0
        self.observation_buffer = deque(self.initial_obs, maxlen=3)

        if self.randomizeCameraParamOnReset:
            self.camera_settings["angle"]["pitch"] = np.random.uniform(8, 12)
            self.camera_settings["angle"]["roll"] = np.random.uniform(-2, 2)

            # print(self.camera_settings["angle"]["pitch"])
            # print(self.camera_settings["angle"]["roll"])
        tmpSettings = deepcopy(self.camera_settings)
        tmpSettings["resolution"] = (640, 480)
        self.sim = Simulator(cameraSettings=tmpSettings)

        tmpMapSettings = deepcopy(self.map_parameters)
        # tmpMapSettings["loops"] = random.randint(1,3)
        # tmpMapSettings["expansions"] = random.randint(5,7)
        # tmpMapSettings["complications"] = random.randint(4,6)

        self.sim = Simulator(cameraSettings=tmpSettings)

        self.sim.start(
            # mapSeed=random.randint(0,1000),
            mapSeed='real',
            mapParameters=tmpMapSettings,
            carParameters=self.car_parameters,
        )

        # where, facing = self.sim.RealSense.parent.ackermann.pose()
        # initial_img = self.sim.RealSense.camera.getImage(where, facing)
        # print(f"initial_img.shape: {initial_img.shape}")
        # observation = self.preprocess_img(initial_img)
        # print(f"observation.shape: {observation.shape}")
        # self.observation_buffer.append(observation)

        if self.use3imgBuffer:
            return np.dstack(self.initial_obs)  # reward, done, info can't be included
        else:
            return self.initial_obs