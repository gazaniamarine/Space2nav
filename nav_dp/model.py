"""Official standalone NavDP policy ("navdp-cross-modal" checkpoint).

Switched from the InternVLA-N1-embedded DAT variant (which needs
checkpoints/navdp_extracted.pth -- not present anywhere on this machine) to
this one because its weights (navdp-cross-modal.ckpt, 543MB, obtained via
the official InternRobotics/NavDP request form) are the only NavDP weights
that actually exist on disk here. Architecture mirror of internnav's
`NavDPNet` built only from the vendored backbone module.

Hyperparameters (probed from the state dict): memory_size=8,
predict_size=24, token_dim=384, temporal_depth=16, heads=8, DDPM 10 train
timesteps.

Input conventions:
  images: (B, memory_size, 224, 224, 3) float, RGB / 255.0
  depths: (B, 1, 224, 224, 1) float, meters (current frame only), clipped
          to [0.1, 5.0]
  goal_point: (3,) [x fwd, y, z] meters, clipped x:[0,10] y,z:[-10,10]

predict_pointgoal/predict_nogoal add critic-based top-k ranking on top of
the checkpoint's own sample_pointgoal/sample_nogoal (which return every
sampled trajectory unranked) -- same ranking convention as the DAT variant
this replaced, so stagewise/executor.py's call sites didn't need to change.
There is no trained pixel-goal encoder exposed here (the checkpoint's own
pixel_encoder expects a 4ch RGB+mask crop, not a bare normalized pixel --
not wired up yet); use predict_pointgoal with a pixel+depth-projected 3D
point instead (see stagewise/executor.py).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .backbone import (
    ImageGoalBackbone,
    LearnablePositionalEncoding,
    PixelGoalBackbone,
    RGBDBackbone,
    SinusoidalPosEmb,
)

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "navdp-cross-modal.ckpt")
_DEFAULT_DA_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "depth_anything_v2_vits.pth"
)


class NavDPStandalone(nn.Module):
    def __init__(
        self,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        pixel_channel=4,
        dropout=0.1,
        da_checkpoint=_DEFAULT_DA_CKPT,
        device="cuda:0",
    ):
        super().__init__()
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.token_dim = token_dim
        self.dropout = dropout
        self.input_dtype = torch.float32
        self._device = torch.device(device)

        self.rgbd_encoder = RGBDBackbone(
            image_size, token_dim, memory_size=memory_size, finetune=False,
            checkpoint=da_checkpoint, device=device,
        )
        self.pixel_encoder = PixelGoalBackbone(
            image_size, token_dim, pixel_channel=pixel_channel, device=device
        )
        self.image_encoder = ImageGoalBackbone(image_size, token_dim, device=device)
        self.point_encoder = nn.Linear(3, token_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=temporal_depth)
        self.input_embed = nn.Linear(3, token_dim)

        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.drop = nn.Dropout(dropout)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", clip_sample=True,
            prediction_type="epsilon",
        )

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float("-inf"))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        )
        self.cond_critic_mask = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask[:, 0:4] = float("-inf")

        self.pixel_aux_head = nn.Linear(token_dim, 3)
        self.image_aux_head = nn.Linear(token_dim, 3)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, weights_path=_DEFAULT_WEIGHTS, device="cuda:0", **kwargs):
        model = cls(device=device, **kwargs)
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "state_dict" in state:
            state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith("rgbd_encoder.rgb_model.")]
        if real_missing or unexpected:
            print(f"[NavDP] missing({len(real_missing)})={real_missing[:6]} "
                  f"unexpected({len(unexpected)})={unexpected[:6]}")
        else:
            print("[NavDP] all weights matched")
        model = model.to(device)
        model.tgt_mask = model.tgt_mask.to(device)
        model.cond_critic_mask = model.cond_critic_mask.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    def _predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self._device)).unsqueeze(1)
        cond = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1)
        cond = cond + self.cond_pos_embed(cond)
        cond = cond.repeat(action_embeds.shape[0], 1, 1)
        inp = action_embeds + self.out_pos_embed(action_embeds)
        out = self.decoder(tgt=inp, memory=cond, tgt_mask=self.tgt_mask)
        out = self.layernorm(out)
        return self.action_head(out)

    def _predict_critic(self, trajectories, rgbd_embed):
        repeat_rgbd = rgbd_embed.repeat(trajectories.shape[0], 1, 1)
        nogoal = torch.zeros_like(repeat_rgbd[:, 0:1])
        act = self.input_embed(trajectories)
        act = act + self.out_pos_embed(act)
        cond = torch.cat([nogoal, nogoal, nogoal, nogoal, repeat_rgbd], dim=1)
        cond = cond + self.cond_pos_embed(cond)
        out = self.decoder(tgt=act, memory=cond, memory_mask=self.cond_critic_mask)
        out = self.layernorm(out)
        return self.critic_head(out.mean(dim=1))[:, 0]

    def _denoise(self, goal_embed, rgbd_embed, sample_num):
        naction = torch.randn((sample_num, self.predict_size, 3), device=self._device)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self._predict_noise(naction, k.unsqueeze(0), goal_embed, rgbd_embed)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        return naction

    def _rank(self, naction, rgbd_embed, topk=8):
        critic = self._predict_critic(naction, rgbd_embed).float()
        trajs = torch.cumsum(naction.float() / 4.0, dim=1)
        best = trajs[(-critic).argsort()[:topk]]
        worst = trajs[critic.argsort()[:topk]]
        return best, worst, critic

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_pointgoal(self, goal_point, images, depths, sample_num=32, topk=8):
        """goal_point: (3,) or (1,3) [x fwd, y, z] meters, robot frame."""
        goal = np.asarray(goal_point, dtype=np.float32).reshape(1, 3).clip(-10, 10)
        goal[:, 0] = goal[:, 0].clip(0, 10)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self._device)
        rgbd_embed = self.rgbd_encoder(images.float().to(self._device), depths.float().to(self._device))
        goal_embed = self.point_encoder(goal).unsqueeze(1)
        naction = self._denoise(goal_embed, rgbd_embed, sample_num)
        return self._rank(naction, rgbd_embed, topk)

    @torch.no_grad()
    def predict_nogoal(self, images, depths, sample_num=32, topk=8):
        rgbd_embed = self.rgbd_encoder(images.float().to(self._device), depths.float().to(self._device))
        goal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
        naction = self._denoise(goal_embed, rgbd_embed, sample_num)
        return self._rank(naction, rgbd_embed, topk)


def trajectory_to_vw(trajectory, kp=1.0, max_lin=0.5, max_ang=0.5):
    """Last waypoint of a ranked trajectory -> (linear, angular) velocity."""
    subgoal = trajectory[-1]
    linear = float(np.clip(kp * np.linalg.norm(subgoal[:2]), 0.0, max_lin))
    angular = float(np.clip(kp * subgoal[2], -max_ang, max_ang))
    return linear, angular
