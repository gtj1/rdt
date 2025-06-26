import sys
sys.path.append("~/Desktop/RoboticsDiffusionTransformer")

import argparse
import yaml
import numpy as np
import torch
import time

from itertools import product
from PIL import Image
from collections import deque
from data.core import (RobotController, RobotCommand, CommandQueue, 
                       RecordQueue, CameraCollector, ImageColor)

from scripts.aubo_model import create_model, RoboticDiffusionTransformerModel

from typing import Any, TypedDict, Literal
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R


CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

@dataclass
class Argument:
    # config: dict[str, Any]
    pretrained_model_name_or_path: str
    lang_embeddings_path: str

    max_publish_step: int = 10000
    ctrl_freq: int = 25
    chunk_size: int = 64
    arm_steps_length: tuple[float, ...] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2)
    config_path: str = 'configs/base.yaml'

class InferenceConfig(TypedDict):
    episode_len: int
    state_dim: int
    chunk_size: int
    camera_names: list[str]


# See https://arxiv.org/pdf/1812.07035
RotationRepresentation = np.ndarray[tuple[Literal[6],], np.dtype[np.float_]]

# joints position (6)
# end effector position (3)
# end_effector angle (6)
# gripper (1)
StateVector = np.ndarray[tuple[Literal[6+3+6+1],], np.dtype[np.float_]]
StateVectorChunk = np.ndarray[tuple[int, Literal[6+3+6+1]], np.dtype[np.float_]]
UnifiedVectorChunk = np.ndarray[tuple[int, Literal[128]], np.dtype[np.float_]]

ImageObservation = dict[str, ImageColor | None]

class Observation(TypedDict):
    qpos: StateVector
    images: ImageObservation


lang_embeddings: torch.Tensor
observation_window: deque[Observation]

class Rate:
    frequency: float
    time_step: float
    last_time: float

    def __init__(self, frequency: float):
        self.frequency = frequency
        self.time_step = 1 / frequency
        self.last_time = 0

    def sleep(self):
        dt = time.time() - self.last_time
        if dt < self.time_step:
            time.sleep(self.time_step - dt)
        self.last_time = time.time()

    

# Initialize the model
def make_policy(args: Argument):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model

# Interpolate the actions to make the robot move smoothly
def interpolate_action(
    args: Argument, 
    prev_action: StateVector, 
    cur_action: StateVector
):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def get_config(args: Argument):
    config = InferenceConfig(
        episode_len=args.max_publish_step,
        state_dim=14,
        chunk_size=args.chunk_size,
        camera_names=CAMERA_NAMES,
    )
    return config


def get_observation(
    args: Argument, 
    config: InferenceConfig,
    controller: RobotController,
    cameras: CameraCollector
) -> Observation:
    assert controller.record_queue is not None
    if not controller.record_queue.empty():
        raise ValueError("Unexpected unread message")
    controller.execute(
        RobotCommand(command_type='record', arm_state=None, gripper_state=None)
    )
    record = controller.record_queue.get()
    state = record['state']
    arm_state = state['arm_state']
    gripper_state = state['gripper_state']
    assert arm_state is not None and gripper_state is not None
    joint = arm_state['joint_position']
    xyz, rpy = arm_state['end_effector_pose']
    rotation = R.from_euler('xyz', list(rpy), degrees=False)
    rotation_matrix = rotation.as_matrix()
    ortho6d = rotation_matrix[:, :2].transpose().flatten()
    gripper_position = [gripper_state['position'] / 255.0]

    qpos: StateVector = np.concatenate([joint, xyz, ortho6d, gripper_position])
    frame = cameras.shot()
    
    images: ImageObservation = {
        config['camera_names'][0]: frame['usb_image'],
        config['camera_names'][1]: frame['rgbd_image'],
        config['camera_names'][2]: None
    }

    return Observation(qpos=qpos, images=images)


def update_observation_window(
    args: Argument, 
    config: InferenceConfig, 
    controller: RobotController,
    cameras: CameraCollector
):
    observation = get_observation(args, config, controller, cameras)
    observation_window.append(observation)


def inference_fn(
    args: Argument,
    config: InferenceConfig,
    policy: RoboticDiffusionTransformerModel,
    t: int
) -> StateVectorChunk:
    image_arrs = [
        observation_window[i]['images'][j] for i, j in\
              product((-2, -1), config['camera_names'])
    ]
    images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
    proprio = torch.tensor(observation_window[-1]['qpos'], device='cuda').float().unsqueeze(0)

    actions = policy.step(
        proprio=proprio,
        images=images,
        text_embeds=lang_embeddings
    )

    return actions.squeeze(0).cpu().numpy()

def init_globals(
    args: Argument,
    config: InferenceConfig
):
    global lang_embeddings
    lang_dict: dict[str, Any] = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict['embeddings']

    global observation_window
    observation_window = deque(maxlen=2)
    observation_window.append(Observation(
        qpos=np.zeros(16),
        images={
            name: None for name in config['camera_names']
        }
    ))

def model_inference(
    args: Argument, 
    config: InferenceConfig,
    controller: RobotController,
    cameras: CameraCollector
) -> None:
    init_globals(args, config)
    policy = make_policy(args)

    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    command = controller.create_move_command(
        end_effector_pose=((-0.36, 0.085, 0.30), (3.14, -1.0, -3.14)), gripper_position=0
    )
    input("Press Enter to continue ... ")

    rate = Rate(args.ctrl_freq)

    with torch.inference_mode():
        while True:
            t = 0
            action_buffer: StateVectorChunk = np.zeros([chunk_size])

            print(f"timestep: {t}")

            while t < max_publish_step:
                update_observation_window(args, config, controller, cameras)

                if t % chunk_size == 0:
                    t1 = time.time()
                    action_buffer = inference_fn(args, config, policy, t)
                    t2 = time.time()
                    print(f"Inference time: {t2 - t1:.4f} seconds")

                action: StateVector = action_buffer[t % chunk_size]
                joint_position = tuple(action[0:6].tolist())
                gripper_position = int(action[-1:].item() * 255)

                command = controller.create_move_command(
                    joint_position=joint_position,
                    gripper_position=gripper_position
                )
                controller.execute(command)

                rate.sleep()
                t += 1

                print("Published Step", t)



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return Argument(**args.__dict__)


def main():
    args = get_arguments()
    robot_controller = RobotController(
        command_queue=CommandQueue(),
        record_queue=RecordQueue()
    )
    robot_controller.connect()
    cameras = CameraCollector()
    config = get_config(args)

    time.sleep(3)
    try:
        model_inference(args, config, robot_controller, cameras)
    except KeyboardInterrupt:
        robot_controller.release()
        cameras.release()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     robot_controller.release()
    #     cameras.release()
    finally:
        print("Exit!")


if __name__ == "__main__":
    main()