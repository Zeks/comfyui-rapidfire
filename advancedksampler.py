
from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import folder_paths

import ast
from pathlib import Path
from importlib import import_module
import os
import sys
import copy
import subprocess
import json
import psutil

import re
import nodes as native
from .adv_encode import advanced_encode #, advanced_encode_XL

from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_gits import GITSScheduler
import comfy_extras.nodes_model_advanced

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



class LoraTagLoader:
    def __init__(self):
        self.loaded_lora = None
        self.tag_pattern = "\<[0-9a-zA-Z\:\_\-\.\s\/\(\)\\\\]+\>"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "text": ("STRING", {"multiline": True}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, text):
        # print(f"\nLoraTagLoader input text: { text }")

        founds = re.findall(self.tag_pattern, text)
        # print(f"\nfoound lora tags: { founds }")

        if len(founds) < 1:
            return (model, clip, text)

        model_lora = model
        clip_lora = clip
        
        lora_files = folder_paths.get_filename_list("loras")
        for f in founds:
            tag = f[1:-1]
            pak = tag.split(":")
            type = pak[0]
            if type != 'lora':
                continue
            name = None
            if len(pak) > 1 and len(pak[1]) > 0:
                name = pak[1]
            else:
                continue
            wModel = wClip = 0
            try:
                if len(pak) > 2 and len(pak[2]) > 0:
                    wModel = float(pak[2])
                    wClip = wModel
                if len(pak) > 3 and len(pak[3]) > 0:
                    wClip = float(pak[3])
            except ValueError:
                continue
            if name == None:
                continue
            lora_name = None
            for lora_file in lora_files:
                if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
                    lora_name = lora_file
                    break
            if lora_name == None:
                print(f"bypassed lora tag: { (type, name, wModel, wClip) } >> { lora_name }")
                continue
            print(f"detected lora tag: { (type, name, wModel, wClip) } >> { lora_name }")

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    temp = self.loaded_lora
                    self.loaded_lora = None
                    del temp

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)

            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, wModel, wClip)

        plain_prompt = re.sub(self.tag_pattern, "", text)
        return (model_lora, clip_lora, plain_prompt)


class AdvancedCLIPTextEncodeWithBreak:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "clip": ("CLIP", ),
            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "weight_interpretation": (["comfy", "A1111", "compel", "comfy++" ,"down_weight"],),
            }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning/advanced"

    def _encode(self, clip, text, token_normalization, weight_interpretation):
        embeddings_final, pooled = advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0)
        return ([[embeddings_final, {"pooled_output": pooled}]], )

    def encode(self, clip, text, token_normalization, weight_interpretation):
        prompts = re.split(r"\s*\bBREAK\b\s*", text) 
        # encode first prompt fragment
        prompt = prompts.pop(0)
        # print(f"prompt: {prompt}")
        out = self._encode(clip, prompt, token_normalization, weight_interpretation)
        # encode and concatenate the rest of the prompt
        for prompt in prompts:
            # print(f"prompt: {prompt}")
            cond_to = self._encode(clip, prompt, token_normalization, weight_interpretation)
            out = native.ConditioningConcat.concat(self, cond_to[0], out[0])
        return out



class TwoModelAdvancedKsampler:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model1": ("MODEL",),
                "clip1": ("CLIP", ),
                "model2": ("MODEL",),
                "clip2": ("CLIP", ),
                "positive": ("STRING",),
                "negative": ("STRING",),
                "lora_name": ("STRING",),
                "rescaled_steps": ("INT",{"default": 8, "min": 0, "max": 100},),
                "rescale_multiplier": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1,
                        "step": 0.05,
                        "round": 0.01,
                    },
                ),
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
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++" ,"down_weight"],),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, 
        model1, clip1, 
        model2, clip2,
        positive, negative, lora_name,
        rescaled_steps,  rescale_multiplier,
        total_steps_original, total_steps_shift,
        noise_seed,
        sampler_name, scheduler,
        starting_cfg, cfg_shift,
        steps1_start, steps1_end, start_steps2_shift,
        token_normalization,weight_interpretation,
        latent_image,
        **kwargs):
            
        model1, clip1, text1 = LoraTagLoader().load_lora(model1, clip1, lora_name)
        model2, clip2, text2 = LoraTagLoader().load_lora(model2, clip2, lora_name)
        
        positive_conditioning1 = AdvancedCLIPTextEncodeWithBreak().encode(clip1, positive, token_normalization, weight_interpretation)
        positive_conditioning2 = AdvancedCLIPTextEncodeWithBreak().encode(clip2, positive, token_normalization, weight_interpretation)
        negative_conditioning1 = AdvancedCLIPTextEncodeWithBreak().encode(clip1, negative, token_normalization, weight_interpretation)
        negative_conditioning2 = AdvancedCLIPTextEncodeWithBreak().encode(clip2, negative, token_normalization, weight_interpretation)
        
        # Debug print all input parameters
        print("\n=== TwoModelAdvancedKsampler Debug Information ===")
        print(f"Model1: {type(model1)}")
        print(f"Model2: {type(model2)}")

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
        samples = []
        if rescaled_steps > 0:
            CFGR = comfy_extras.nodes_model_advanced.RescaleCFG()
            patched_model = CFGR.patch(model1, rescale_multiplier)[0]
            print("Attepmpting rescaled CFG pass")
            try:
                samples = KSamplerAdvanced().sample(
                    patched_model, 
                    "enable",  # add_noise
                    noise_seed, 
                    total_steps_original, 
                    starting_cfg, 
                    sampler_name, 
                    scheduler,
                    positive_conditioning1[0], 
                    negative_conditioning1[0], 
                    latent_image, 
                    steps1_start, 
                    rescaled_steps,
                    "enable"  # return_with_leftover_noise
                )
                print("Rescaled CFG pass completed successfully")
                print(f"Output latent shape: {samples[0]['samples'].shape}")
            except Exception as e:
                print(f"\n!!! ERROR in first sampling pass !!!")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                raise e        
        
        
        try:
            actual_steps = (steps1_start + rescaled_steps) if rescaled_steps > 0 else steps1_start
            samples = KSamplerAdvanced().sample(
                model1, 
                "disable",  # add_noise
                noise_seed, 
                total_steps_original, 
                starting_cfg, 
                sampler_name, 
                scheduler,
                positive_conditioning1[0], 
                negative_conditioning1[0], 
                samples[0], 
                actual_steps, 
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
                positive_conditioning2[0], 
                negative_conditioning2[0], 
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