from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import ast
from pathlib import Path
from importlib import import_module
import os
import sys
import copy
import subprocess
import json
import psutil

from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_gits import GITSScheduler

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
from nodes import LatentUpscaleBy, KSampler, KSamplerAdvanced, VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled, \
    ImageScaleBy, CLIPSetLastLayer, CLIPTextEncode, ControlNetLoader, ControlNetApply, ControlNetApplyAdvanced, \
    PreviewImage, MAX_RESOLUTION
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats
sys.path.remove(comfy_dir)

from comfy import samplers
# Append custom_nodes_dir to sys.path
sys.path.append(custom_nodes_dir)


class TwoModelAdvancedKsampler:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model1": ("MODEL",),
                "positive1": ("CONDITIONING",),
                "negative1": ("CONDITIONING",),
                "model2": ("MODEL",),
                "positive2": ("CONDITIONING",),
                "negative2": ("CONDITIONING",),
                "total_steps_original": ("INT",{"default": 25, "min": 0, "max": 100},),
                "total_steps_shift": ("INT",{"default": 0, "min": -50, "max": 100},),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "starting_cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "cfg_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "steps1_start": ("INT", {"default": 0, "min": 0, "max": 100}),
                "steps1_end": ("INT", {"default": 0, "min": 0, "max": 100}),
                "start_steps2_shift": ("INT", {"default": 0, "min": -20, "max": 100}),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, 
        model1, positive1, negative1,
        model2, positive2, negative2, 
        total_steps_original, total_steps_shift,
        noise_seed,
        sampler_name, scheduler,
        starting_cfg, cfg_shift,
        steps1_start, steps1_end, start_steps2_shift,
        latent_image,
        **kwargs):
        
        # Debug print all input parameters
        print("\n=== TwoModelAdvancedKsampler Debug Information ===")
        print(f"Model1: {type(model1)}")
        print(f"Model2: {type(model2)}")
        print(f"Positive1: {len(positive1)} elements")
        print(f"Negative1: {len(negative1)} elements")
        print(f"Positive2: {len(positive2)} elements")
        print(f"Negative2: {len(negative2)} elements")
        print(f"Total Steps Original: {total_steps_original}")
        print(f"Total Steps Shift: {total_steps_shift} (New total: {total_steps_original + total_steps_shift})")
        print(f"Noise Seed: {noise_seed}")
        print(f"Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"Starting CFG: {starting_cfg}, CFG Shift: {cfg_shift} (New CFG: {starting_cfg + cfg_shift})")
        print(f"Model1 Steps Range: {steps1_start} to {steps1_end}")
        print(f"Model2 Start Step Shift: {start_steps2_shift} (Absolute start step: {steps1_end + start_steps2_shift})")
        print(f"Latent Image Shape: {latent_image['samples'].shape}")
        
        # First sampling pass
        print("\n=== Starting First Sampling Pass ===")
        print(f"Using Model1 from step {steps1_start} to {steps1_end}")
        
        try:
            samples = KSamplerAdvanced().sample(
                model1, 
                "enable",  # add_noise
                noise_seed, 
                total_steps_original, 
                starting_cfg, 
                sampler_name, 
                scheduler,
                positive1, 
                negative1, 
                latent_image, 
                steps1_start, 
                steps1_end,
                "enable"  # return_with_leftover_noise
            )
            print("First sampling pass completed successfully")
            print(f"Output latent shape: {samples[0]['samples'].shape}")
        except Exception as e:
            print(f"\n!!! ERROR in first sampling pass !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise e
        
        # Second sampling pass
        print("\n=== Starting Second Sampling Pass ===")
        new_total_steps = total_steps_original + total_steps_shift
        new_cfg = starting_cfg + cfg_shift
        model2_start_step = steps1_end + start_steps2_shift
        model2_end_step = new_total_steps
        
        print(f"Using Model2 from step {model2_start_step} to {model2_end_step}")
        print(f"New total steps: {new_total_steps}")
        print(f"New CFG scale: {new_cfg}")
        
        try:
            final_samples = KSamplerAdvanced().sample(
                model2, 
                "disable",  # add_noise (disabled for second pass)
                noise_seed, 
                new_total_steps, 
                new_cfg, 
                sampler_name, 
                scheduler,
                positive2, 
                negative2, 
                samples[0], 
                model2_start_step, 
                model2_end_step,
                "enable"  # return_with_leftover_noise
            )
            print("Second sampling pass completed successfully")
            print(f"Final output latent shape: {final_samples[0]['samples'].shape}")
        except Exception as e:
            print(f"\n!!! ERROR in second sampling pass !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise e
        
        print("\n=== Sampling Completed Successfully ===")
        return final_samples