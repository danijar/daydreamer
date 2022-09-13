import dataclasses
import enum
import random
import sys
import time
import traceback
from typing import Any, Dict, Tuple

import cv2
import dcargs
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

import embodied
from embodied.envs.spacemouse import SpaceMouse


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.001)
        self.last = time.time()


@dataclasses.dataclass
class EnvConfig:
    control_rate_hz: float = 2
    with_camera: bool = True
    debug_cam_vis: bool = False
    use_real: bool = True


class SpheroEnv:
    LOW_WHITE_THRESH = (94, 87, 83)
    HIGH_WHITE_THRESH = (129, 255, 171)
    RC_MIN = (0, 0)
    RC_MAX = (64, 64)
    MAX_CONTROL = 70  # should be between 0 and 255
    ARENA_MIN = np.array((193, 67))
    ARENA_MAX = np.array((480, 370))
    GOAL_POS = (0.825, 0.165)

    X_ENV_CROP = slice(80, -30)
    SUCCESS_THRESHOLD = 0.1

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

        if cfg.use_real:
            self._toy = scanner.find_toy()
            self._api = SpheroEduAPI(self._toy).__enter__()
            self._api.set_stabilization(False)
        else:
            self._toy = None
            if not self.cfg.debug_cam_vis:
                return

        if self.cfg.with_camera:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)

            self.rate = Rate(cfg.control_rate_hz)

            self.last_pos = None
            self._reset(move=False)

            if self.cfg.debug_cam_vis:
                while True:
                    image, depth = self.get_frames()

                    # im = Image.fromarray(image)
                    # im.save("ball.jpg")
                    # import ipdb; ipdb.set_trace();

                    pos, white_mask, _ = self.get_ball_pos(image)
                    # print(pos, self.get_reward())
                    pos = pos * (self.ARENA_MAX - self.ARENA_MIN) + self.ARENA_MIN
                    depth = np.repeat(depth, 3, -1)
                    # mask_3 = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
                    # white_mask = np.repeat(white_mask[:, :, None], 3, -1)
                    # import ipdb; ipdb.set_trace();
                    # cv2.circle(mask_3, pos.astype(np.int), 50, (0, 0, 255))
                    mask = np.repeat(white_mask[..., None], 3, -1)
                    cv2.circle(image, pos.astype(np.int), 50, (0, 255, 0))
                    cv2.circle(mask, pos.astype(np.int), 50, (0, 255, 0))

                    cv2.imshow(
                        "img",
                        np.concatenate([image[:, self.X_ENV_CROP, ::-1], mask], 1),
                    )

                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots()
                    # ax.hist(depth.ravel(), bins=100)
                    # plt.show()

                    # image = cv2.applyColorMap(255 - depth, cv2.COLORMAP_VIRIDIS)
                    # cv2.imshow("img", depth)

                    cv2.waitKey(1)

        self.clear_error_states()

    def __len__(self):
        # Return positive integer for batched envs.
        return 0

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.with_camera:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
            image = color_image
            depth = depth_image[:, :, None]
            # Map depth to used range for our robot setup.
            depth = depth.astype(np.float32) / 255
            nearest = 0.050
            farthest = 0.120
            depth = (depth - nearest) / (farthest - nearest)
            depth = (255 * np.clip(depth, 0, 1)).astype(np.uint8)
        else:
            image = np.zeros((64, 64, 3))
            depth = np.zeros((64, 64, 1))
        return image, depth

    def __del__(self) -> None:
        if self.cfg.use_real:
            self._api.__exit__()

    @property
    def act_space(self) -> Dict[str, embodied.Space]:
        return {
            "action": embodied.Space(np.float32, (2,), low=-1, high=1),
            "reset": embodied.Space(np.bool),
        }

    @property
    def obs_space(self) -> Dict[str, embodied.Space]:
        return {
            "image": embodied.Space(np.uint8, (64, 64, 3)),
            # "depth": embodied.Space(np.uint8, (64, 64, 1)),
            "state": embodied.Space(np.float32, (2,)),
            "goal": embodied.Space(np.float32, (2,)),
            "reward": embodied.Space(np.float32),
            "log_success": embodied.Space(np.uint8),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    def get_ball_pos(self, image: np.ndarray) -> np.ndarray:
        image = cv2.GaussianBlur(image, (15, 15), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.LOW_WHITE_THRESH, self.HIGH_WHITE_THRESH)
        white_mask = cv2.erode(white_mask, None, iterations=2)
        white_mask = cv2.dilate(white_mask, None, iterations=2)

        if (white_mask == 0).all():
            return self.last_pos, white_mask, False

        # white_mask /= white_mask.sum(0, keepdims=True)
        # white_mask /= white_mask.sum(1, keepdims=True)

        # x = (
        #     np.arange(0, 640)[None, :] *
        #     (white_mask / white_mask.sum(1, keepdims=True)).sum()
        #
        # y = np.dot(
        #     np.arange(0, 480)[None, :],
        #     white_mask / white_mask.sum(1, keepdims=True))

        normalize = lambda x: x / x.sum()
        x = np.dot(np.arange(0, 640), normalize(white_mask.mean(0)))
        y = np.dot(np.arange(0, 480), normalize(white_mask.mean(1)))

        pos = np.array([x, y])

        pos = (pos - self.ARENA_MIN) / (self.ARENA_MAX - self.ARENA_MIN)

        self.last_pos = pos
        return pos, white_mask, True

    def get_reward(self, curr_obs: Dict[str, Any]) -> float:
        """Provides the enviroment reward.

        A subclass should implement this method.

        Args:
            curr_obs (Dict[str, Any]): Observation dict.

        Returns:
            float: reward
        """
        ball_pos, _, _ = self.get_ball_pos(self.get_frames()[0])
        reward = -np.linalg.norm(ball_pos - np.array(self.goal))
        # print(reward)
        return reward

    def get_obs(self, is_first: bool = False) -> Dict[str, Any]:
        color_image, depth_image = self.get_frames()
        ball_pos, _, _ = self.get_ball_pos(color_image)

        # change observations to be within reasonable values
        size = (64, 64)
        obs = dict(
            image=cv2.resize(color_image[:, self.X_ENV_CROP, :], size)[:, :, ::-1],
            # depth=cv2.resize(depth_image, size),
            # state=state,
            goal=np.array(self.goal),
            is_last=False,
            is_terminal=False,
            is_first=is_first,
        )
        rew = self.get_reward(color_image)
        obs["reward"] = float(rew)
        obs["log_success"] = np.array(rew > -self.SUCCESS_THRESHOLD, dtype=np.uint8)
        return obs

    def clear_error_states(self):
        print("TODO")

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if action["reset"]:
            return self._reset()

        action = action["action"]
        self._api.raw_motor(
            int(action[0] * self.MAX_CONTROL),
            int(action[1] * self.MAX_CONTROL),
            duration=1 / self.cfg.control_rate_hz,
        )

        self.rate.sleep()
        obs = self.get_obs(is_first=False)
        return obs

    def _reset(self, move: bool = True) -> Dict[str, Any]:
        # x = np.random.uniform(0.1, 0.9)
        # y = np.random.uniform(0.1, 0.9)
        # self.goal = (x, y)
        print("RESET!")
        self.goal = self.GOAL_POS
        while not self.get_ball_pos(self.get_frames()[0])[2]:
            print("Waiting for you to put the ball into the arena...")
            time.sleep(1)
        if move:
            for _ in range(5):
                action = np.random.choice([-1, 1], 2)
                self._api.raw_motor(
                    int(action[0] * 100), int(action[1] * 100), duration=1
                )
            # wait for robot to settle
            time.sleep(4)
        obs = self.get_obs(is_first=True)
        print("Reset done!")
        return obs

    def render(self) -> np.ndarray:
        """TODO"""
        raise NotImplementedError

    def close(self) -> None:
        """Nothing to close."""


class UnsafeTestAgent:
    def __init__(self, env_config: EnvConfig) -> None:
        self._cfg = env_config

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        del obs  # not used

        # randomize direction of action
        control_action = np.random.uniform(-1, 1, 2)
        return {"action": control_action, "reset": False}


def main(env_config: EnvConfig) -> None:
    # select an env
    env = SpheroEnv(env_config)

    # select an agent
    agent = UnsafeTestAgent(env_config)
    print(f"==> Running the environment with {agent}")

    obs = env._reset()
    while True:
        try:
            action = agent.act(obs)
        except IndexError:
            break
        t = time.time()
        action["reset"] = False
        obs = env.step(action)
        reward = obs["reward"]
        if np.abs(reward) > 0.5:
            print(f"Step time : {time.time() - t}, reward: {reward}")

    obs = env._reset()
    print("==> Finishing to run the environment")


if __name__ == "__main__":
    # toy = scanner.find_toy()
    # with SpheroEduAPI(toy) as api:
    #     while True:
    #         action = np.random.uniform(-1, 1, 2)
    #         action = (
    #             np.random.choice([-1, 1], 2)
    #         )
    #         api.raw_motor(int(action[0] * 255), int(action[1] * 255), duration=4)
    main(dcargs.parse(EnvConfig))
