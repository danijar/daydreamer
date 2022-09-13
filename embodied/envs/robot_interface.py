import dataclasses
import enum
import socket
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import dcargs
import numpy as np
import pyrealsense2 as rs

import embodied
from embodied.envs.spacemouse import SpaceMouse


class RobotType(enum.Enum):
    XARM = 1
    UR5 = 2

    def joints(self) -> int:
        if self == RobotType.UR5:
            return 6
        elif self == RobotType.XARM:
            return 7
        else:
            raise NotImplementedError(f"RobotType: {self}")


class Task(enum.Enum):
    PICKPLACE = 1
    SWEEP = 2


class UR5SimpleRobotWrapper:

    # x <--|
    #      |
    #      v
    #      y

    VEL = 8.0
    VEL_Z = 1.0
    ACC = 2.0
    GRIPPER_OPEN = 5
    GRIPPER_CLOSE = 230

    Y_ADJ = 0.08
    X_ADJ = 0.04

    LEFT_XY_MIN = (-0.125, -0.64)
    LEFT_XY_MAX = (0.048, -0.36)
    LEFT_SAFE_XY_MIN = (LEFT_XY_MIN[0] + X_ADJ + 0.06, LEFT_XY_MIN[1] + Y_ADJ)
    LEFT_SAFE_XY_MAX = (LEFT_XY_MAX[0] - X_ADJ, LEFT_XY_MAX[1] - Y_ADJ)

    RIGHT_XY_MIN = (-0.455, -0.64)
    RIGHT_XY_MAX = (-0.285, -0.36)
    RIGHT_SAFE_XY_MIN = (RIGHT_XY_MIN[0] + X_ADJ, RIGHT_XY_MIN[1] + Y_ADJ)
    RIGHT_SAFE_XY_MAX = (RIGHT_XY_MAX[1] - X_ADJ, RIGHT_XY_MAX[1] - Y_ADJ)

    Z_TABLE = -0.010
    Z_HOVER = 0.12
    Z_RESET = 0.18
    AXIS = 0

    def __init__(self):
        import urx
        from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
        [ print("hello") for _ in range(10)]

        robot_ip: str = "172.22.22.3"
        self.robot = urx.Robot(robot_ip)
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.robot, force=0)
        self.gripper_read_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gripper_read_socket.connect(("172.22.22.3", 63352))

        self._set_gripper_position(self.GRIPPER_OPEN)
        self._gripper_state_open = True  # open

    def get_gripper_pos(self) -> float:
        self.gripper_read_socket.sendall(b"GET POS\n")
        data = self.gripper_read_socket.recv(2 ** 10)
        gripper_pos = int(data[4:-1])
        assert 0 <= gripper_pos <= 255

        normalized_gripper_pos = (gripper_pos - self.GRIPPER_OPEN) / (
            self.GRIPPER_CLOSE - self.GRIPPER_OPEN
        )
        return normalized_gripper_pos

    def _set_gripper_position(self, pos: int) -> None:
        assert 0 <= pos <= 255
        self.robotiqgrip.gripper_action(pos)

    def close_gripper(self) -> None:
        if self._gripper_state_open:
            self._set_gripper_position(self.GRIPPER_CLOSE)
        self._gripper_state_open = False

    def open_gripper(self) -> None:
        if not self._gripper_state_open:
            self._set_gripper_position(self.GRIPPER_OPEN)
        self._gripper_state_open = True

    def __del__(self):
        self.robot.close()
        self.gripper_read_socket.close()

    def get_robot_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        gripper_pos = self.get_gripper_pos()
        servo_angle = self.robot.getj()
        cart_pos = self.robot.getl()
        return (
            np.array([gripper_pos], np.float32),
            np.array(servo_angle, np.float32),
            np.array(cart_pos, np.float32),
        )

    def set_position(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        wait: bool = True,
        acc: Optional[float] = None,
    ) -> None:
        if z is None:
            z = self.robot.get_pos().z
        assert z is not None

        if acc is None:
            acc = self.ACC

        self.robot.movel(
            (x, y, z, -2.2213, -2.2213, 0),
            acc=acc,
            vel=self.VEL,
            threshold=0.01,
            wait=wait,
        )

    def set_z(self, z: float, wait: bool = True) -> None:
        p = self.robot.get_pos()
        p.z = z
        self.robot.set_pos(p, acc=self.ACC, vel=self.VEL_Z, threshold=0.01, wait=wait)


class XArmSimpleRobotWrapper:
    VEL = 800
    VEL_Z = 80
    GRIPPER_OPEN = 800
    GRIPPER_CLOSE = 0

    # BBOX CONSTANTS
    #      x
    #      ^
    #      |
    # y <--|
    X_ADJ = 0.08
    Y_ADJ = 0.04

    LEFT_XY_MIN = (0.252, 0.085)
    LEFT_XY_MAX = (0.523, 0.175)
    LEFT_SAFE_XY_MIN = (0.252 + X_ADJ, 0.085 + Y_ADJ)
    LEFT_SAFE_XY_MAX = (0.523 - X_ADJ, 0.175 - Y_ADJ)

    RIGHT_XY_MIN = (0.252, -0.170)
    RIGHT_XY_MAX = (0.523, -0.075)
    RIGHT_SAFE_XY_MIN = (0.252 + X_ADJ, -0.170 + Y_ADJ)
    RIGHT_SAFE_XY_MAX = (0.523 - X_ADJ, -0.075 - Y_ADJ - 0.035)

    Z_TABLE = 0.182
    Z_HOVER = 0.290
    Z_RESET = 0.450

    AXIS = 1

    def __init__(self):
        from xarm.wrapper import XArmAPI

        self.robot = XArmAPI("192.168.1.233")

        self._set_gripper_position(self.GRIPPER_OPEN)
        self._gripper_state_open = True  # open

    def get_gripper_pos(self) -> float:
        code, gripper_pos = self.robot.get_gripper_position()
        while code != 0 or gripper_pos is None:
            print(f"Error code {code} in get_gripper_position(). {gripper_pos}")
            self.clear_error_states()
            code, gripper_pos = self.robot.get_gripper_position()

        normalized_gripper_pos = (gripper_pos - self.GRIPPER_OPEN) / (
            self.GRIPPER_CLOSE - self.GRIPPER_OPEN
        )
        return normalized_gripper_pos

    def _set_gripper_position(self, pos: int) -> None:
        self.robot.set_gripper_position(pos, wait=True)
        while self.robot.get_is_moving():
            time.sleep(0.01)

    def close_gripper(self) -> None:
        if self._gripper_state_open:
            self._set_gripper_position(self.GRIPPER_CLOSE)
        self._gripper_state_open = False

    def open_gripper(self) -> None:
        if not self._gripper_state_open:
            self._set_gripper_position(self.GRIPPER_OPEN)
        self._gripper_state_open = True

    def __del__(self) -> None:
        self.robot.disconnect()

    def clear_error_states(self):
        self.robot.clean_error()
        self.robot.motion_enable(True)
        self.robot.set_mode(0)
        self.robot.set_state(state=0)
        self.robot.set_gripper_enable(True)
        self.robot.set_gripper_speed(10000)

    def get_robot_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        while self.robot.get_is_moving():
            time.sleep(0.01)

        gripper_pos = self.get_gripper_pos()

        code, servo_angle = self.robot.get_servo_angle(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_servo_angle().")
            self.clear_error_states()
            code, servo_angle = self.robot.get_servo_angle(is_radian=True)

        code, cart_pos = self.robot.get_position(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_position().")
            self.clear_error_states()
            code, cart_pos = self.robot.get_position(is_radian=True)

        cart_pos = np.array(cart_pos)
        cart_pos[:3] /= 1000
        return (
            np.array([gripper_pos], np.float32),
            np.array(servo_angle, np.float32),
            np.array(cart_pos, np.float32),
        )

    def set_position(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        wait: bool = True,
        acc: Optional[float] = None,
    ) -> None:
        self.robot.set_position(
            x=1000 * x,
            y=1000 * y,
            z=None if z is None else 1000 * z,
            roll=-180,
            pitch=0,
            yaw=0,
            wait=wait,
            speed=self.VEL,
        )
        while self.robot.get_is_moving():
            time.sleep(0.01)

    def set_z(self, z: float, wait: bool = True) -> None:
        self.robot.set_position(z=1000 * z, wait=wait, speed=self.VEL_Z)
        while self.robot.get_is_moving():
            time.sleep(0.01)


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
    max_delta_m: float = 0.04  # max displacement for the arm per time step
    control_rate_hz: float = 20
    with_camera: bool = True
    debug_cam_vis: bool = False
    use_real: bool = True
    robot_type: RobotType = RobotType.XARM
    enable_z: bool = True
    task: Task = Task.PICKPLACE


class BaseEnv:
    def __init__(self, cfg: EnvConfig):
        if cfg.task == Task.SWEEP:
            assert cfg.enable_z, "z control must be enabled for sweeping"

        if cfg.task == Task.SWEEP:
            raise NotImplementedError()  # TODO

        self.cfg = cfg
        self._arm: Union[XArmSimpleRobotWrapper, UR5SimpleRobotWrapper]
        if cfg.use_real:
            if self.cfg.robot_type == RobotType.XARM:
                self._arm = XArmSimpleRobotWrapper()
            elif self.cfg.robot_type == RobotType.UR5:
                self._arm = UR5SimpleRobotWrapper()
            else:
                raise NotImplementedError(f"arm: {self.cfg.robot_type} not implemented")
        else:
            self._arm = None  # type: ignore
            if not self.cfg.debug_cam_vis:
                return
        self.rate = Rate(cfg.control_rate_hz)

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
            # if self.cfg.debug_cam_vis:
            if self.cfg.debug_cam_vis:
                while True:
                    image = self.get_frames()[0][:, :, ::-1]
                    depth = np.repeat(self.get_frames()[1], 3, -1)
                    cv2.imshow("img", np.concatenate([image, depth], 1))

                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots()
                    # ax.hist(depth.ravel(), bins=100)
                    # plt.show()

                    # image = cv2.applyColorMap(255 - depth, cv2.COLORMAP_VIRIDIS)
                    # cv2.imshow("img", depth)

                    cv2.waitKey(1)

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

            if self.cfg.robot_type == RobotType.UR5:
                depth_image = depth_image[135:-100, 120:-170]
                # depth_image = depth_image[120:-80, 150:-190]
                color_image = color_image[20:-30, 40:-55]
                farthest = 0.180
            else:
                farthest = 0.120

            if self.cfg.debug_cam_vis:
                img_size = (480, 480)
            else:
                img_size = (64, 64)
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)[:, :, None]
            # Map depth to used range for our robot setup.
            depth = depth.astype(np.float32) / 255
            nearest = 0.050
            depth = (depth - nearest) / (farthest - nearest)
            depth = (255 * np.clip(depth, 0, 1)).astype(np.uint8)

        else:
            image = np.zeros((64, 64, 3))
            depth = np.zeros((64, 64, 1))
        return image, depth

    def __del__(self) -> None:
        del self._arm

    @property
    def obs_space(self) -> Dict[str, embodied.Space]:
        return {
            "image": embodied.Space(np.uint8, (64, 64, 3)),
            "depth": embodied.Space(np.uint8, (64, 64, 1)),
            "cartesian_position": embodied.Space(np.float32, (6,)),
            "joint_positions": embodied.Space(
                np.float32, (self.cfg.robot_type.joints(),)
            ),
            "gripper_pos": embodied.Space(np.float32, (1,)),
            "gripper_side": embodied.Space(np.float32, (3,)),
            "grasped_side": embodied.Space(np.float32, (3,)),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    def get_reward(self, curr_obs: Dict[str, Any]) -> float:
        """Provide the enviroment reward.

        A subclass should implement this method.

        Args:
            curr_obs (Dict[str, Any]): Observation dict.

        Returns:
            float: reward
        """
        raise NotImplementedError

    def get_obs(
        self,
        robot_in_safe_state: bool,
        is_first: bool = False,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        color_image, depth_image = self.get_frames()

        # change observations to be within reasonable values
        gripper_pos, servo_angle, cart_pos = self._arm.get_robot_state()
        grasped_side_one_hot = {
            Side.OTHER: [1, 0, 0],
            Side.LEFT: [0, 1, 0],
            Side.RIGHT: [0, 0, 1],
        }
        gripper_side = self.arm_side()
        obs = dict(
            image=color_image,
            depth=depth_image,
            cartesian_position=cart_pos,
            joint_positions=servo_angle,
            gripper_pos=gripper_pos,
            gripper_side=np.array(grasped_side_one_hot[self.grasped_bin], np.float32),
            grasped_side=np.array(grasped_side_one_hot[gripper_side], np.float32),
            is_last=False,
            is_terminal=False,
        )

        if reward is None:
            if robot_in_safe_state:
                obs["reward"] = float(self.get_reward(obs))
            else:
                obs["reward"] = float(0)
        else:
            obs["reward"] = reward

        obs["is_first"] = is_first
        return obs

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _reset(self) -> Dict[str, Any]:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        """TODO."""
        raise NotImplementedError

    def close(self) -> None:
        """Nothing to close."""


def check_grasped_object_ur(gripper_pos: float) -> bool:
    if 0.015 < gripper_pos and gripper_pos < 0.985:
        return True
    return False


class Side(enum.Enum):
    LEFT = 1
    RIGHT = 2
    OTHER = 3


DEBUG_DELTA = 0.0
# DEBUG_DELTA = 0.2


class PickPlace(BaseEnv):  # GraspRewardEnv
    COUNTDOWN_STEPS = 3

    def random_xy_grid(self, side: Side) -> Tuple[float, float]:
        if side == Side.LEFT:
            x = np.random.uniform(
                self._arm.LEFT_SAFE_XY_MIN[0], self._arm.LEFT_SAFE_XY_MAX[0]
            )
            y = np.random.uniform(
                self._arm.LEFT_SAFE_XY_MIN[1], self._arm.LEFT_SAFE_XY_MAX[1]
            )
        elif side == Side.RIGHT:
            x = np.random.uniform(
                self._arm.RIGHT_SAFE_XY_MIN[0], self._arm.RIGHT_SAFE_XY_MAX[0]
            )
            y = np.random.uniform(
                self._arm.RIGHT_SAFE_XY_MIN[1], self._arm.RIGHT_SAFE_XY_MAX[1]
            )
        else:
            raise NotImplementedError(f"Got side: {side} ")

        x = np.round(x / self.cfg.max_delta_m) * self.cfg.max_delta_m
        y = np.round(y / self.cfg.max_delta_m) * self.cfg.max_delta_m

        if side == Side.LEFT:
            x = np.clip(x, self._arm.LEFT_SAFE_XY_MIN[0], self._arm.LEFT_SAFE_XY_MAX[0])
            y = np.clip(y, self._arm.LEFT_SAFE_XY_MIN[1], self._arm.LEFT_SAFE_XY_MAX[1])
        elif side == Side.RIGHT:
            x = np.clip(
                x, self._arm.RIGHT_SAFE_XY_MIN[0], self._arm.RIGHT_SAFE_XY_MAX[0]
            )
            y = np.clip(
                y, self._arm.RIGHT_SAFE_XY_MIN[1], self._arm.RIGHT_SAFE_XY_MAX[1]
            )
        else:
            raise NotImplementedError(f"Got side: {side} ")
        return x, y

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        if cfg.use_real:
            self.grasped_object = False
            self.grasped_bin = Side.OTHER
            self._ball_side = Side.LEFT
            self._arm.set_z(self._arm.Z_RESET)
            x, y = self.random_xy_grid(self._ball_side)
            self._arm.set_position(x, y, self._arm.Z_RESET)
            self._reset()

    def _print_debug_info(self) -> None:
        print(f"Right Bound Min: {self._arm.RIGHT_XY_MIN}")
        print(f"Right Bound Max: {self._arm.RIGHT_XY_MAX}")
        print(f"Left  Bound Min: {self._arm.LEFT_XY_MIN}")
        print(f"Left  Bound Max: {self._arm.LEFT_XY_MAX}")
        traceback.print_stack()
        stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
        print(stacktrace)

    def arm_side(self, margin: float = -0.002) -> Side:
        _, _, cart_pos = self._arm.get_robot_state()

        pos = np.array(cart_pos)[:2]
        if (np.array(self._arm.RIGHT_XY_MIN) + margin <= pos).all() and (
            np.array(self._arm.RIGHT_XY_MAX) - margin >= pos
        ).all():
            return Side.RIGHT
        elif (np.array(self._arm.LEFT_XY_MIN) + margin <= pos).all() and (
            np.array(self._arm.LEFT_XY_MAX) - margin >= pos
        ).all():
            return Side.LEFT
        else:
            print("ARM NOT ON EITHER SIDE")
            print(cart_pos[:2])
            self._print_debug_info()
            self._reset()
            return self.arm_side()

    def current_bounds(self) -> Tuple[np.ndarray, np.ndarray, float]:

        side = self.arm_side()
        if side == Side.LEFT:
            if self.is_hover():
                return (
                    np.array(self._arm.LEFT_SAFE_XY_MIN),
                    np.array(self._arm.LEFT_SAFE_XY_MAX),
                    self._arm.Z_HOVER,
                )
            else:
                return (
                    np.array(self._arm.LEFT_XY_MIN),
                    np.array(self._arm.LEFT_XY_MAX),
                    self._arm.Z_TABLE,
                )
        elif side == Side.RIGHT:
            if self.is_hover():
                return (
                    np.array(self._arm.RIGHT_SAFE_XY_MIN),
                    np.array(self._arm.RIGHT_SAFE_XY_MAX),
                    self._arm.Z_HOVER,
                )
            else:
                return (
                    np.array(self._arm.RIGHT_XY_MIN),
                    np.array(self._arm.RIGHT_XY_MAX),
                    self._arm.Z_TABLE,
                )
        else:
            raise NotImplementedError

    @property
    def act_space(self) -> Dict[str, embodied.Space]:
        if self.cfg.enable_z:
            return {
                "action": embodied.Space(np.int64, (), 0, 6),
            }
        else:
            return {
                "action": embodied.Space(np.int64, (), 0, 5),
            }

    def is_hover(self) -> bool:
        _, _, cart_pos = self._arm.get_robot_state()
        return cart_pos[2] > (self._arm.Z_HOVER + self._arm.Z_TABLE) / 2

    def compute_arm_position(self, control_action: np.ndarray) -> np.ndarray:
        """Convert control action to TCP homogeneous transform.

        Args:
            env_config (EnvConfig): The environment configuration.
            control_action (np.ndarray, shape=self.control_shape()): control_action
            (should be values between -1 and 1, following the dm_control convention)
            curr_pose (np.ndarray, shape=(6, )): the current robot pose

        Returns:
            np.ndarray, shape=(6, ): The target pose.
        """
        control_action = np.clip(control_action, -1, 1) * self.cfg.max_delta_m
        assert control_action.shape == (2,), control_action

        _, _, cart_pos = self._arm.get_robot_state()
        target_pose = np.array(cart_pos)
        target_pose[:2] = target_pose[:2] + control_action

        xy_min, xy_max, z_loc = self.current_bounds()

        target_pose[:2] = (
            np.round(target_pose[:2] / self.cfg.max_delta_m)
        ) * self.cfg.max_delta_m

        desired_pose = np.copy(target_pose)
        target_pose[:2] = np.clip(target_pose[:2], xy_min, xy_max)

        if self.grasped_object and self.is_hover():
            side = self.arm_side()
            # cross the middle if holding object
            if (
                side == Side.LEFT
                and desired_pose[self._arm.AXIS] + 0.01 < target_pose[self._arm.AXIS]
            ):
                target_pose[:2] = np.clip(
                    target_pose[:2],
                    self._arm.RIGHT_SAFE_XY_MIN,
                    self._arm.RIGHT_SAFE_XY_MAX,
                )
            if (
                side == Side.RIGHT
                and desired_pose[self._arm.AXIS] - 0.01 > target_pose[self._arm.AXIS]
            ):
                target_pose[:2] = np.clip(
                    target_pose[:2],
                    self._arm.LEFT_SAFE_XY_MIN,
                    self._arm.LEFT_SAFE_XY_MAX,
                )

        target_pose[2] = z_loc
        if control_action[0] == 0:
            target_pose[0] = cart_pos[0]
        if control_action[1] == 0:
            target_pose[1] = cart_pos[1]
        return target_pose

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if action["reset"]:
            if action.get("manual_resume", False):
                return self.get_obs(robot_in_safe_state=True, is_first=True)
            else:
                return self._reset()


        if action["action"] < 4:
            pos_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))[action["action"]]
            xyzrpy = self.compute_arm_position(np.array(pos_delta))
            self._arm.set_position(xyzrpy[0], xyzrpy[1])

        elif action["action"] == 4:  # close
            if self._arm._gripper_state_open:
                self._arm.close_gripper()
            else:
                self._arm.open_gripper()

        elif action["action"] == 5:  # close
            arm_side: Side = self.arm_side()
            _, _, cart_pos = self._arm.get_robot_state()
            is_hover = cart_pos[2] > (self._arm.Z_HOVER + self._arm.Z_TABLE) / 2
            if is_hover:
                self._arm.set_z(self._arm.Z_TABLE)
            elif self.grasped_object:
                _, _, cart_pos = self._arm.get_robot_state()
                if arm_side == Side.LEFT:
                    cart_pos[:2] = np.clip(
                        cart_pos[:2],
                        self._arm.LEFT_SAFE_XY_MIN,
                        self._arm.LEFT_SAFE_XY_MAX,
                    )
                else:
                    cart_pos[:2] = np.clip(
                        cart_pos[:2],
                        self._arm.RIGHT_SAFE_XY_MIN,
                        self._arm.RIGHT_SAFE_XY_MAX,
                    )
                if self.cfg.enable_z:
                    self._arm.set_position(cart_pos[0], cart_pos[1], self._arm.Z_TABLE)
                    self._arm.set_position(cart_pos[0], cart_pos[1], self._arm.Z_HOVER)
            else:
                # no object so no op
                pass
        else:
            raise NotImplementedError(action)

        self.rate.sleep()

        obs = self.get_obs(
            robot_in_safe_state=True, is_first=False
        )  # TODO: check safe state
        if obs["reward"] != 0:
            obs = self.get_obs(
                robot_in_safe_state=True, is_first=False, reward=obs["reward"]
            )

        if action.get("manual_pause", False):
            self._arm.open_gripper()
        return obs

    def _reset(self) -> Dict[str, Any]:
        if self.grasped_object:
            # move to random pos in bin where object was grasped
            x, y = self.random_xy_grid(self._ball_side)
            self._arm.set_position(x, y, self._arm.Z_HOVER)

        self._arm.open_gripper()
        self.grasped_bin = Side.OTHER
        self.grasped_object = False

        if self._ball_side == Side.LEFT:
            xyz_min, xyz_max = self._arm.LEFT_XY_MIN, self._arm.LEFT_XY_MAX
        elif self._ball_side == Side.RIGHT:
            xyz_min, xyz_max = self._arm.RIGHT_XY_MIN, self._arm.RIGHT_XY_MAX
        else:
            raise NotImplementedError(f"ball side={self._ball_side}")

        if self.cfg.robot_type == RobotType.UR5:
            # get ball out of corners
            for corner_x, corner_y in ([1, 0], [0, 0], [0, 1], [1, 1]):
                if corner_x == 0:
                    x = xyz_min[0]
                else:
                    x = xyz_max[0]
                if corner_y == 0:
                    y = xyz_min[1]
                else:
                    y = xyz_max[1]
                time.sleep(2)
                self._arm.set_position(x, y, self._arm.Z_TABLE, acc=0.6)

        self._arm.open_gripper()
        x, y = self.random_xy_grid(self._ball_side)
        self._arm.set_position(x, y, self._arm.Z_TABLE, acc=0.6)
        time.sleep(1)  # wait for scene to settle after reset

        obs = self.get_obs(robot_in_safe_state=True, is_first=True)
        return obs

    def get_reward(self, curr_obs: Dict[str, Any]) -> float:
        grasped_old = self.grasped_object
        grasped_new = check_grasped_object_ur(curr_obs["gripper_pos"])
        self.grasped_object = grasped_new

        arm_side: Side = self.arm_side()
        if grasped_old and grasped_new and arm_side != self.grasped_bin:
            # let go of ball
            self._arm.open_gripper()
            self._ball_side = arm_side
            self.grasped_object = False

            # move arm down
            self._arm.set_z(self._arm.Z_TABLE)

            # move arm to random location on success
            x, y = self.random_xy_grid(self._ball_side)
            self._arm.set_position(x, y, self._arm.Z_TABLE)
            return 10

        if not grasped_old and not grasped_new:  # not holding it
            return 0

        if not grasped_old and grasped_new:  # grasped it
            self.grasped_bin = arm_side
            assert self.grasped_bin != Side.OTHER
            _, _, cart_pos = self._arm.get_robot_state()
            if arm_side == Side.LEFT:
                cart_pos[:2] = np.clip(
                    cart_pos[:2], self._arm.LEFT_SAFE_XY_MIN, self._arm.LEFT_SAFE_XY_MAX
                )
            else:
                cart_pos[:2] = np.clip(
                    cart_pos[:2],
                    self._arm.RIGHT_SAFE_XY_MIN,
                    self._arm.RIGHT_SAFE_XY_MAX,
                )
            if not self.cfg.enable_z:
                self._arm.set_position(cart_pos[0], cart_pos[1], self._arm.Z_TABLE)
                self._arm.set_position(cart_pos[0], cart_pos[1], self._arm.Z_HOVER)
            return 1

        if grasped_old and not grasped_new:  # dropped it
            assert arm_side == self.grasped_bin
            rew = -1
            self._arm.set_z(self._arm.Z_TABLE)
            self.grasped_bin = Side.OTHER
            return rew

        if grasped_old and grasped_new:  # holding it
            return 0

        raise NotImplementedError


class SpaceMouseAgent:
    def __init__(self, env_config: EnvConfig) -> None:
        self._cfg = env_config
        self._mouse = SpaceMouse()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        while True:
            input_position = np.zeros((2,))
            if np.linalg.norm(self._mouse.input_pos) > 10:
                # print(self._mouse.input_pos)
                input_position = (
                    np.asarray(
                        [
                            self._mouse.input_pos[1],
                            self._mouse.input_pos[0],
                            -self._mouse.input_pos[2],
                        ]
                    )
                    / 350
                )
                input_position = input_position[:2]

            if np.abs(input_position[0]) > np.abs(input_position[1]):
                if input_position[0] > 0.2:
                    return {"action": 0}
                elif input_position[0] < -0.2:
                    return {"action": 1}
            else:
                if input_position[1] > 0.2:
                    return {"action": 2}
                elif input_position[1] < -0.2:
                    return {"action": 3}

            if self._mouse.input_button1:
                return {"action": 4}
            if self._mouse.input_button0 and self._cfg.enable_z:
                return {"action": 5}

            time.sleep(0.005)  # wait for an action


class UnsafeTestAgent:
    def __init__(self, env_config: EnvConfig) -> None:
        self._cfg = env_config

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        del obs  # not used

        # randomize direction of action
        control_action = np.random.randint(0, 6)
        return {"action": control_action, "reset": False}


def main(env_config: EnvConfig) -> None:
    print(env_config)
    # select an env
    # env = KeyRewardEnv(env_config)
    # env = CornerRewardEnv(env_config)
    env = PickPlace(cfg=env_config)

    # select an agent
    # agent = UnsafeTestAgent(env_config)
    agent = SpaceMouseAgent(env_config)
    print(f"==> Running the environment with {agent}")

    obs = env._reset()
    count = 0
    while True:
        count += 1
        if count % 100 == 0:
            env._reset()
        try:
            action = agent.act(obs)  # type: ignore
        except IndexError:
            break
        t = time.time()
        action["reset"] = False
        start = time.time()
        obs = env.step(action)
        print(f"action={action}, {time.time() - start}")
        reward = obs["reward"]
        if np.abs(reward) > 0.5:
            print(f"Step time : {time.time() - t}, reward: {reward}")
        if obs["is_last"]:
            env.step({"reset": True})
    print("==> Finishing to run the environment")


if __name__ == "__main__":
    main(dcargs.parse(EnvConfig))
