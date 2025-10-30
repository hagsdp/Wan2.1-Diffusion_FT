# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import time

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 profile: bool = False):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            # Notes on performance (where time is spent):
            # - Each iteration of the following `for` loop (the tqdm loop) performs the
            #   heavy work of diffusion sampling. Concretely it runs:
            #     1) (possible) model device transfer (`self.model.to(self.device)`) — expensive if moving
            #        a large model between CPU/GPU per-step.
            #     2) model forward for conditional context (heavy; transformer conv/attn blocks)
            #     3) model forward for unconditional (heavy; often identical cost to (2))
            #     4) guidance combine (cheap elementwise ops)
            #     5) scheduler.step(...) (mostly arithmetic, negligible vs model forward)
            #   The profiler below optionally measures these timings so you can identify
            #   whether the model forwards or device transfers dominate runtime.

            # Lightweight profiling containers (only if `profile=True`). We store per-step
            # durations for each major sub-step and report averages at the end.
            profile_stats = None
            if profile:
                profile_stats = {
                    'model_to_device': [],
                    'forward_cond': [],
                    'forward_uncond': [],
                    'guidance': [],
                    'scheduler': [],
                    'latents_update': [],
                    'vae_decode': [],
                }
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                # 1) ensure model is on the target device — this may be a no-op if already on device.
                if profile:
                    t0 = time.perf_counter()
                # keep the existing behavior (call .to every step) but measure it when profiling
                self.model.to(self.device)
                if profile:
                    # sync to get accurate GPU time measurements
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    profile_stats['model_to_device'].append(time.perf_counter() - t0)

                # 2/3) batched conditional + unconditional forward (heavy)
                # Build a batch of two identical latents and two contexts (cond, uncond)
                # so we only call the model once per timestep. This reduces overhead
                # and is safe because the model is stateless across these two calls.
                if profile:
                    t0 = time.perf_counter()

                # duplicate latent tensor for batch-size=2
                x0 = latent_model_input[0]
                x_batch = [x0, x0]

                # create timestep batch: repeat the single timestep twice
                t_batch = timestep.repeat(2)

                # prepare context batch: (cond, uncond)
                # context objects are typically lists of tensors; extract the per-sample tensor
                if isinstance(arg_c['context'], (list, tuple)):
                    ctx_c = arg_c['context'][0]
                else:
                    ctx_c = arg_c['context']
                if isinstance(arg_null['context'], (list, tuple)):
                    ctx_n = arg_null['context'][0]
                else:
                    ctx_n = arg_null['context']
                context_batch = [ctx_c, ctx_n]

                # single batched forward -> returns a list of two outputs
                outs = self.model(x_batch, t=t_batch, context=context_batch, seq_len=seq_len)
                noise_pred_cond = outs[0]
                noise_pred_uncond = outs[1]

                # respect offload_model semantics (move outputs to CPU if requested)
                if offload_model:
                    noise_pred_cond = noise_pred_cond.to(torch.device('cpu'))
                    noise_pred_uncond = noise_pred_uncond.to(torch.device('cpu'))
                else:
                    noise_pred_cond = noise_pred_cond.to(self.device)
                    noise_pred_uncond = noise_pred_uncond.to(self.device)

                if profile:
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    t_elapsed = time.perf_counter() - t0
                    # record batched time and also append to the per-role lists for compatibility
                    profile_stats.setdefault('forward_batched', []).append(t_elapsed)
                    profile_stats['forward_cond'].append(t_elapsed)
                    profile_stats['forward_uncond'].append(t_elapsed)

                # 4) guidance combine (cheap)
                if profile:
                    t0 = time.perf_counter()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                if profile:
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    profile_stats['guidance'].append(time.perf_counter() - t0)

                # 5) scheduler update: uses the model output to compute the previous latent
                if profile:
                    t0 = time.perf_counter()
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                if profile:
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    profile_stats['scheduler'].append(time.perf_counter() - t0)

                # update latents (very cheap)
                if profile:
                    t0 = time.perf_counter()
                latents = [temp_x0.squeeze(0)]
                if profile:
                    profile_stats['latents_update'].append(time.perf_counter() - t0)

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            # measure VAE decode time when profiling
            if profile:
                t0 = time.perf_counter()
            if self.rank == 0:
                videos = self.vae.decode(x0)
            if profile:
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                profile_stats['vae_decode'].append(time.perf_counter() - t0)

            # report profiling summary (on rank 0)
            if profile and self.rank == 0:
                def _avg(l):
                    return float(sum(l) / len(l)) if len(l) > 0 else 0.0

                logging.info("Profiling summary (per-step averages in seconds):")
                logging.info(f" model_to_device: {_avg(profile_stats['model_to_device']):.6f}")
                logging.info(f" forward_cond: {_avg(profile_stats['forward_cond']):.6f}")
                logging.info(f" forward_uncond: {_avg(profile_stats['forward_uncond']):.6f}")
                logging.info(f" guidance: {_avg(profile_stats['guidance']):.6f}")
                logging.info(f" scheduler: {_avg(profile_stats['scheduler']):.6f}")
                logging.info(f" latents_update: {_avg(profile_stats['latents_update']):.6f}")
                logging.info(f" vae_decode (total time): {sum(profile_stats['vae_decode']):.6f}")

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
