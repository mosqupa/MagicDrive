from typing import Any, Dict, Optional, Tuple, Union, List
import logging

import random
import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange

from .output_cls import BEVControlNetOutput
from .unet_addon_rawbox import BEVControlNetModel

class VirtualImageControlNet(BEVControlNetModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        camera_param: torch.Tensor,  # BEV
        bboxes_3d_data: Dict[str, Any],  # BEV
        encoder_hidden_states: torch.Tensor,
        virtual_image_embedding: torch.FloatTensor, # new_add 
        controlnet_cond: torch.FloatTensor,
        encoder_hidden_states_uncond: torch.Tensor = None,  # BEV
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[BEVControlNetOutput, Tuple]:
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(
                f"unknown `controlnet_conditioning_channel_order`: {channel_order}"
            )

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. camera
        N_cam = camera_param.shape[1]
        camera_emb = self._embed_camera(camera_param)
        # (B, N_cam, max_len + 1, dim=768)
        encoder_hidden_states_with_cam = self.add_cam_states(
            encoder_hidden_states, camera_emb
        )
        # we may drop the condition during training, but not drop controlnet
        if (self.drop_cond_ratio > 0.0 and self.training):
            if encoder_hidden_states_uncond is not None:
                encoder_hidden_states_with_cam, uncond_mask = self._random_use_uncond_cam(
                    encoder_hidden_states_with_cam, encoder_hidden_states_uncond)
            controlnet_cond = controlnet_cond.type(self.dtype)
            controlnet_cond = self._random_use_uncond_map(controlnet_cond)
        else:
            uncond_mask = None

        # 0.5. bbox embeddings
        # bboxes data should follow the format of (B, N_cam or 1, max_len, ...)
        # for each view
        if bboxes_3d_data is not None:
            bbox_embedder_kwargs = {}
            for k, v in bboxes_3d_data.items():
                bbox_embedder_kwargs[k] = v.clone()
            if self.drop_cam_with_box and uncond_mask is not None:
                _, n_box = bboxes_3d_data["bboxes"].shape[:2]
                if n_box != N_cam:
                    assert n_box == 1, "either N_cam or 1."
                    for k in bboxes_3d_data.keys():
                        ori_v = rearrange(
                            bbox_embedder_kwargs[k], 'b n ... -> (b n) ...')
                        new_v = repeat(ori_v, 'b ... -> b n ...', n=N_cam)
                        bbox_embedder_kwargs[k] = new_v
                # here we set mask for dropped boxes to all zero
                masks = bbox_embedder_kwargs['masks']
                masks[uncond_mask > 0] = 0
            # original flow
            b_box, n_box = bbox_embedder_kwargs["bboxes"].shape[:2]
            for k in bboxes_3d_data.keys():
                bbox_embedder_kwargs[k] = rearrange(
                    bbox_embedder_kwargs[k], 'b n ... -> (b n) ...')
            bbox_emb = self.bbox_embedder(**bbox_embedder_kwargs)
            if n_box != N_cam:
                # n_box should be 1: all views share the same set of bboxes, we repeat
                bbox_emb = repeat(bbox_emb, 'b ... -> b n ...', n=N_cam)
            else:
                # each view already has its set of bboxes
                bbox_emb = rearrange(bbox_emb, '(b n) ... -> b n ...', n=N_cam)
            encoder_hidden_states_with_cam = torch.cat([
                encoder_hidden_states_with_cam, bbox_emb
            ], dim=2)


        #################### add virtual image embedding #####################
        # 0.6. virtual image embedding
        if virtual_image_embedding is not None:
            if len(virtual_image_embedding.shape) == 1:
                # (768,)
                virtual_image_embedding = virtual_image_embedding.unsqueeze(0)
            elif len(virtual_image_embedding.shape) == 2:
                # (B, 768)
                pass
            else:
                raise ValueError(
                    f"Unexpected shape for virtual_image_embedding: {virtual_image_embedding.shape}"
                )
            virtual_image_embedding = repeat(
                virtual_image_embedding, 'b ... -> b n ...', n=N_cam).unsqueeze(2)
            encoder_hidden_states_with_cam = torch.cat([
                encoder_hidden_states_with_cam, virtual_image_embedding
            ], dim=2)
        ######################################################################


        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps],
                dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(sample.shape[0])

        timesteps = timesteps.reshape(-1)  # time_proj can only take 1-D input
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # BEV: we remap data to have (B n) as batch size
        sample = rearrange(sample, 'b n ... -> (b n) ...')
        encoder_hidden_states_with_cam = rearrange(
            encoder_hidden_states_with_cam, 'b n ... -> (b n) ...')
        if len(emb) < len(sample):
            emb = repeat(emb, 'b ... -> (b repeat) ...', repeat=N_cam)
        controlnet_cond = repeat(
            controlnet_cond, 'b ... -> (b repeat) ...', repeat=N_cam)

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        sample += controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_with_cam,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states_with_cam,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(
            down_block_res_samples, self.controlnet_down_blocks
        ):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode:
            scales = torch.logspace(
                -1, 0, len(down_block_res_samples) + 1
            )  # 0.1 to 1.0
            scales *= conditioning_scale
            down_block_res_samples = [
                sample * scale for sample,
                scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample *= scales[-1]  # last one
        else:
            down_block_res_samples = [
                sample * conditioning_scale for sample in down_block_res_samples
            ]
            mid_block_res_sample *= conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True)
                for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(
                mid_block_res_sample, dim=(2, 3), keepdim=True
            )

        if not return_dict:
            return (
                down_block_res_samples,
                mid_block_res_sample,
                encoder_hidden_states_with_cam,
            )

        return BEVControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            encoder_hidden_states_with_cam=encoder_hidden_states_with_cam,
        )