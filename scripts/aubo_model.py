import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower, SiglipImageProcessor
from models.multimodal_encoder.t5_encoder import T5Embedder, T5EncoderModel
from models.rdt_runner import RDTRunner

from typing import Any, Literal
import numpy.typing as npt

from data.core import ImageColor

# The indices that the raw vector should be mapped to in the unified action vector
# AGILEX_STATE_INDICES = [
#     STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
# ] + [
#     STATE_VEC_IDX_MAPPING["left_gripper_open"]
# ] + [
#     STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
# ] + [
#     STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
# ]


AUBO_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"eef_pos_{ax}"] for ax in "xyz"
] + [
    STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"gripper_joint_0_pos"]
]


# Create the RDT model
def create_model(args: Any, **kwargs: Any):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if (
        pretrained is not None 
        and os.path.isfile(pretrained)
    ):
        model.load_pretrained_weights(pretrained)
    return model


class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """

    def __init__(
        self, args: Any, 
        device: str='cuda',
        dtype: torch.dtype=torch.bfloat16,
        image_size: tuple[int, int] | None=None,
        control_frequency: int=25,
        pretrained: str | None=None,
        pretrained_vision_encoder_name_or_path: str | None=None,
        pretrained_text_encoder_name_or_path: str | None=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = torch.device(device)
        self.control_frequency = control_frequency
        self.text_tokenizer, self.text_model = None, None
        # We do not use the text encoder due to limited GPU memory
        use_text_encoder = False
        if use_text_encoder:
            self.text_tokenizer, self.text_model = self.get_text_encoder(
                pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(
            pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy(pretrained)
        
        self.reset()

    def get_policy(self, pretrained: str | None) -> RDTRunner:
        """Initialize the model."""
        # Initialize model with arguments
        if (
            pretrained is None
            or os.path.isfile(pretrained)
        ):
            num_patches = self.vision_model.num_patches

            assert isinstance(num_patches, int)
            img_cond_len: int = (self.args["common"]["img_history_size"] 
                               * self.args["common"]["num_cameras"] 
                               * num_patches)
            
            _model = RDTRunner(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    ("image", (self.args["common"]["img_history_size"], 
                        self.args["common"]["num_cameras"], 
                        -num_patches)),  
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
            )
            
            assert isinstance(_model, RDTRunner)
            
            return _model
        
        else:
            _model = RDTRunner.from_pretrained(pretrained)

            return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path: str | None):
        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path, 
            model_max_length=self.args["dataset"]["tokenizer_max_length"], 
            device=self.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        assert isinstance(text_encoder, T5EncoderModel)
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path: str | None):
        vision_encoder = SiglipVisionTower(
            vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        assert isinstance(image_processor, SiglipImageProcessor)

        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        if self.text_model:
            self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        if self.text_model:
            self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained: str | None=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction: str, device: str="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        if self.text_tokenizer is None or self.text_model is None:
            raise

        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"]
        
        tokens = tokens.to(device).view(1, -1)

        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    def _format_joint_to_state(self, joints: torch.Tensor):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]). 
        """
        # Rescale the gripper to the range of [0, 1]
        # joints = joints / torch.tensor(
        #     [[[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]],
        #     device=joints.device, dtype=joints.dtype
        # )
        
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        # Fill into the unified state vector
        state[:, :, AUBO_STATE_INDICES] = joints
        # state[:, :, AGILEX_STATE_INDICES] = joints
        # Assemble the mask indicating each dimension's availability 
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        # state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        state_elem_mask[:, AUBO_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action: torch.Tensor):
        """
        Unformat the unified action vector into the joint action to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted. 
                ([B, N, 128])
        
        Returns:
            joints (torch.Tensor): The unformatted robot joint action. 
                qpos ([B, N, 14]).
        """
        # action_indices = AGILEX_STATE_INDICES
        # joints = action[:, :, action_indices]
        joints = action[:, :, AUBO_STATE_INDICES]
        
        # Rescale the gripper back to the action range
        # Note that the action range and proprioception range are different
        # for Mobile ALOHA robot
        # joints = joints * torch.tensor(
        #     [[[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]],
        #     device=joints.device, dtype=joints.dtype
        # )
        
        return joints

    @torch.no_grad()
    def step(
        self, proprio: torch.Tensor, images: list[Image.Image | None], 
        text_embeds: torch.Tensor
    ):
        """
        Predict the next action chunk given the 
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, 
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        
        assert isinstance(self.image_processor.image_mean, list)

        # The background image used for padding
        background_color: ImageColor = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        # Preprocess the images by order and encode them
        image_tensor_list = list[torch.Tensor]()

        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            if self.image_size is not None:
                # suspected problem
                # image = transforms.Resize(self.data_args.image_size)(image)
                image = transforms.Resize(self.image_size)(image)
            
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values: list[list[int]] = list(image.getdata())
                average_brightness = sum(
                    sum(pixel) for pixel in pixel_values
                ) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':

                def expand2square(pil_img: Image.Image, background_color: tuple[float, ...]):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                    
                assert isinstance(self.image_processor.image_mean, list)
                image = expand2square(
                    image, tuple(int(x*255) for x in self.image_processor.image_mean))

            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image_tensor)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = (
            states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        )
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory
