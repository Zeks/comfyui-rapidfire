from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import folder_paths
import comfy.model_management

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

import comfy.sd
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


class MultiModelAdvancedKsampler:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": { 
                "used_model_count": ("INT",{"default": 2, "min": 1, "max": 3},),
                "ckpt_name1": (folder_paths.get_filename_list("checkpoints"),),
                "ckpt_name2": (folder_paths.get_filename_list("checkpoints"),),
                "ckpt_name3": (folder_paths.get_filename_list("checkpoints"),),
                "positive": ("STRING", {"multiline": False}),
                "negative": ("STRING", {"multiline": False}),
                "lora_name": ("STRING", {"default": ""}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "rescaled_steps": ("INT",{"default": 8, "min": 0, "max": 100}),
                "rescale_multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1, "step": 0.05}),
                "total_steps_original": ("INT",{"default": 25, "min": 1, "max": 100}),
                "total_steps_shift_second": ("INT",{"default": 0, "min": -50, "max": 100}),
                "total_steps_shift_third": ("INT",{"default": 0, "min": -50, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "starting_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_shift": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 100.0, "step": 0.1}),
                "steps_end_first": ("INT", {"default": 15, "min": 0, "max": 100}),
                "steps_shift_second": ("INT", {"default": 0, "min": -20, "max": 100}),
                "steps_end_second": ("INT", {"default": 0, "min": 0, "max": 100}),
                "steps_shift_third": ("INT", {"default": 0, "min": -20, "max": 100}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++" ,"down_weight"],),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "load_settings": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("LATENT", "settings")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    def __init__(self):
        self.loaded_checkpoints = {}  # Dictionary to store loaded checkpoints by name
        self.current_checkpoints = set()  # Track currently requested checkpoints
        
    def serialize_settings(self, **kwargs):
        """Serialize all input parameters into a human-readable string"""
        used_model_count = kwargs.get("used_model_count", 2)
        
        settings = {
            "used_model_count": used_model_count,
            "ckpt_name1": kwargs.get("ckpt_name1", ""),
            # Only include ckpt_name2 if used_model_count >= 2
            **({"ckpt_name2": kwargs.get("ckpt_name2", "")} if used_model_count >= 2 else {}),
            # Only include ckpt_name3 if used_model_count >= 3
            **({"ckpt_name3": kwargs.get("ckpt_name3", "")} if used_model_count >= 3 else {}),
            "positive": kwargs.get("positive", ""),
            "negative": kwargs.get("negative", ""),
            "lora_name": kwargs.get("lora_name", ""),
            "noise_seed": kwargs.get("noise_seed", 0),
            "rescaled_steps": kwargs.get("rescaled_steps", 8),
            "rescale_multiplier": kwargs.get("rescale_multiplier", 0.7),
            "total_steps_original": kwargs.get("total_steps_original", 25),
            # Only include second model steps if used_model_count >= 2
            **({"total_steps_shift_second": kwargs.get("total_steps_shift_second", 0)} if used_model_count >= 2 else {}),
            # Only include third model steps if used_model_count >= 3
            **({"total_steps_shift_third": kwargs.get("total_steps_shift_third", 0)} if used_model_count >= 3 else {}),
            "sampler_name": kwargs.get("sampler_name", "euler"),
            "scheduler": kwargs.get("scheduler", "normal"),
            "starting_cfg": kwargs.get("starting_cfg", 8.0),
            "cfg_shift": kwargs.get("cfg_shift", 0.0),
            "steps_end_first": kwargs.get("steps_end_first", 15),
            # Only include second model steps if used_model_count >= 2
            **({"steps_shift_second": kwargs.get("steps_shift_second", 0)} if used_model_count >= 2 else {}),
            **({"steps_end_second": kwargs.get("steps_end_second", 0)} if used_model_count >= 2 else {}),
            # Only include third model steps if used_model_count >= 3
            **({"steps_shift_third": kwargs.get("steps_shift_third", 0)} if used_model_count >= 3 else {}),
            "token_normalization": kwargs.get("token_normalization", "none"),
            "weight_interpretation": kwargs.get("weight_interpretation", "comfy"),
        }
        
        return json.dumps(settings, indent=2)
    
    def deserialize_settings(self, settings_str):
        """Deserialize settings string back to a dictionary"""
        if not settings_str.strip():
            return None
        
        try:
            return json.loads(settings_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding settings: {e}")
            return None
    
    def get_effective_settings(self, **kwargs):
        """Get the effective settings, either from load_settings or direct inputs"""
        load_settings = kwargs.get("load_settings", "")
        
        if load_settings and load_settings.strip():
            loaded = self.deserialize_settings(load_settings)
            if loaded:
                # Create a new kwargs with loaded settings, but keep original latent_image
                effective_kwargs = loaded.copy()
                effective_kwargs["latent_image"] = kwargs["latent_image"]
                return effective_kwargs
        
        # If no valid settings to load, use direct inputs
        return kwargs
    
    def purge_unused_checkpoints(self, requested_checkpoints):
        """Purges checkpoints that aren't in the current request"""
        # Convert to set for faster lookups
        requested_set = set(requested_checkpoints)
        
        # Find checkpoints to remove
        to_remove = [name for name in self.loaded_checkpoints 
                    if name not in requested_set]
        
        # Purge unused checkpoints
        for name in to_remove:
            print(f"Purging unused checkpoint: {name}")
            del self.loaded_checkpoints[name]
        
        # Update current checkpoints
        self.current_checkpoints = requested_set

    def load_model_pipeline(self, ckpt_name, lora_name, positive, negative,
                          token_normalization, weight_interpretation):
        """Helper function to load a complete model pipeline"""
        # Check if we already have this checkpoint loaded
        if ckpt_name in self.loaded_checkpoints:
            model, clip, vae = self.loaded_checkpoints[ckpt_name]
            print(f"Using cached checkpoint: {ckpt_name}")
        else:
            # Load base checkpoint if not found in cache
            print(f"Loading new checkpoint: {ckpt_name}")
            out = comfy.sd.load_checkpoint_guess_config(
                folder_paths.get_full_path("checkpoints", ckpt_name),
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            model, clip, vae = out[0], out[1], out[2]
            self.loaded_checkpoints[ckpt_name] = (model, clip, vae)
        
        # Apply LoRA if specified
        if lora_name and lora_name.strip():
            model, clip, _ = LoraTagLoader().load_lora(model, clip, lora_name)
        
        # Generate conditioning
        positive_conditioning = AdvancedCLIPTextEncodeWithBreak().encode(
            clip, positive, token_normalization, weight_interpretation
        )
        
        negative_conditioning = AdvancedCLIPTextEncodeWithBreak().encode(
            clip, negative, token_normalization, weight_interpretation
        )
        
        return {
            'model': model,
            'clip': clip,
            'vae': vae,
            'positive_conditioning': positive_conditioning[0],
            'negative_conditioning': negative_conditioning[0]
        }

    def sample(self, **kwargs):
        # Get effective settings (either from load_settings or direct inputs)
        effective_kwargs = self.get_effective_settings(**kwargs)
        load_settings = kwargs.get("load_settings", "")
        
        # Initialize UI update dictionary
        ui_updates = {}
        
        # If we loaded settings from the string, populate ui_updates
        if load_settings and load_settings.strip():
            loaded_settings = self.deserialize_settings(load_settings)
            if loaded_settings:
                ui_updates = {
                    "used_model_count": (loaded_settings["used_model_count"],),
                    "ckpt_name1": (loaded_settings["ckpt_name1"],),
                    "ckpt_name2": (loaded_settings["ckpt_name2"],),
                    "ckpt_name3": (loaded_settings["ckpt_name3"],),
                    "positive": (loaded_settings["positive"],),
                    "negative": (loaded_settings["negative"],),
                    "lora_name": (loaded_settings["lora_name"],),
                    "noise_seed": (loaded_settings["noise_seed"],),
                    "rescaled_steps": (loaded_settings["rescaled_steps"],),
                    "rescale_multiplier": (loaded_settings["rescale_multiplier"],),
                    "total_steps_original": (loaded_settings["total_steps_original"],),
                    "total_steps_shift_second": (loaded_settings["total_steps_shift_second"],),
                    "total_steps_shift_third": (loaded_settings["total_steps_shift_third"],),
                    "sampler_name": (loaded_settings["sampler_name"],),
                    "scheduler": (loaded_settings["scheduler"],),
                    "starting_cfg": (loaded_settings["starting_cfg"],),
                    "cfg_shift": (loaded_settings["cfg_shift"],),
                    "steps_end_first": (loaded_settings["steps_end_first"],),
                    "steps_shift_second": (loaded_settings["steps_shift_second"],),
                    "steps_end_second": (loaded_settings["steps_end_second"],),
                    "steps_shift_third": (loaded_settings["steps_shift_third"],),
                    "token_normalization": (loaded_settings["token_normalization"],),
                    "weight_interpretation": (loaded_settings["weight_interpretation"],),
                }
        
        # Extract all parameters from effective_kwargs
        used_model_count = effective_kwargs["used_model_count"]
        ckpt_name1 = effective_kwargs["ckpt_name1"]
        ckpt_name2 = effective_kwargs["ckpt_name2"]
        ckpt_name3 = effective_kwargs["ckpt_name3"]
        positive = effective_kwargs["positive"]
        negative = effective_kwargs["negative"]
        lora_name = effective_kwargs["lora_name"]
        noise_seed = effective_kwargs["noise_seed"]
        rescaled_steps = effective_kwargs["rescaled_steps"]
        rescale_multiplier = effective_kwargs["rescale_multiplier"]
        total_steps_original = effective_kwargs["total_steps_original"]
        total_steps_shift_second = effective_kwargs["total_steps_shift_second"]
        total_steps_shift_third = effective_kwargs["total_steps_shift_third"]
        sampler_name = effective_kwargs["sampler_name"]
        scheduler = effective_kwargs["scheduler"]
        starting_cfg = effective_kwargs["starting_cfg"]
        cfg_shift = effective_kwargs["cfg_shift"]
        steps_end_first = effective_kwargs["steps_end_first"]
        steps_shift_second = effective_kwargs["steps_shift_second"]
        steps_end_second = effective_kwargs["steps_end_second"]
        steps_shift_third = effective_kwargs["steps_shift_third"]
        token_normalization = effective_kwargs["token_normalization"]
        weight_interpretation = effective_kwargs["weight_interpretation"]
        latent_image = effective_kwargs["latent_image"]
        
        # Determine which checkpoints are being requested
        requested_checkpoints = [ckpt_name1]
        if used_model_count >= 2:
            requested_checkpoints.append(ckpt_name2)
            
        if used_model_count >= 3:
            requested_checkpoints.append(ckpt_name3)
        
        # Purge checkpoints that aren't being used this run
        self.purge_unused_checkpoints(requested_checkpoints)
        
        # Load all required pipelines
        pipelines = []
        pipelines.append(self.load_model_pipeline(ckpt_name1, lora_name, positive, negative,
                                               token_normalization, weight_interpretation))
        if comfy.model_management.processing_interrupted():
            return []
        
        if used_model_count >= 2:
            pipelines.append(self.load_model_pipeline(ckpt_name2, lora_name, positive, negative,
                                                    token_normalization, weight_interpretation))
        if comfy.model_management.processing_interrupted():
            return []
        if used_model_count >= 3:
            pipelines.append(self.load_model_pipeline(ckpt_name3, lora_name, positive, negative,
                                                    token_normalization, weight_interpretation))

        # First sampling pass
        samples = []
        if rescaled_steps > 0:
            try:
                CFGR = comfy_extras.nodes_model_advanced.RescaleCFG()
                patched_model = CFGR.patch(pipelines[0]['model'], rescale_multiplier)[0]
                print("Attempting rescaled CFG pass")
                
                samples = KSamplerAdvanced().sample(
                    patched_model, 
                    "enable",  # add_noise
                    noise_seed, 
                    total_steps_original, 
                    starting_cfg, 
                    sampler_name, 
                    scheduler,
                    pipelines[0]['positive_conditioning'], 
                    pipelines[0]['negative_conditioning'], 
                    latent_image, 
                    0, 
                    rescaled_steps,
                    "enable"  # return_with_leftover_noise
                )
                print(f"Rescaled CFG pass completed, latent shape: {samples[0]['samples'].shape}")
            except Exception as e:
                print(f"Error in rescaled CFG pass: {str(e)}")
                raise e
        
        if comfy.model_management.processing_interrupted():
            return []
        
        # First model's main pass
        try:
            actual_steps = rescaled_steps if rescaled_steps > 0 else 0
            samples = KSamplerAdvanced().sample(
                pipelines[0]['model'],
                "disable" if rescaled_steps > 0 else "enable",  # add_noise
                noise_seed,
                total_steps_original,
                starting_cfg,
                sampler_name,
                scheduler,
                pipelines[0]['positive_conditioning'],
                pipelines[0]['negative_conditioning'],
                samples[0] if rescaled_steps > 0 else latent_image,
                actual_steps,
                steps_end_first,
                "enable"  # return_with_leftover_noise
            )
            print(f"First model pass completed, latent shape: {samples[0]['samples'].shape}")
        except Exception as e:
            print(f"Error in first model pass: {str(e)}")
            raise e
        
        if comfy.model_management.processing_interrupted():
            return []
        
        # Second model pass if enabled
        if used_model_count >= 2:
            try:
                end_steps = steps_end_second if steps_end_second > 0 else total_steps_original + total_steps_shift_second
                new_cfg = starting_cfg + cfg_shift
                model2_start_step = steps_end_first + steps_shift_second
                new_total_steps = total_steps_original + total_steps_shift_second
                
                samples = KSamplerAdvanced().sample(
                    pipelines[1]['model'],
                    "disable",
                    noise_seed,
                    new_total_steps,
                    new_cfg,
                    sampler_name,
                    scheduler,
                    pipelines[1]['positive_conditioning'],
                    pipelines[1]['negative_conditioning'],
                    samples[0],
                    model2_start_step,
                    end_steps,
                    "enable"
                )
                print(f"Second model pass completed, latent shape: {samples[0]['samples'].shape}")
            except Exception as e:
                print(f"Error in second model pass: {str(e)}")
                raise e
        
        if comfy.model_management.processing_interrupted():
            return []
        
        # Third model pass if enabled
        if used_model_count >= 3:
            try:
                new_total_steps = total_steps_original + total_steps_shift_third + total_steps_shift_second
                new_cfg = starting_cfg + cfg_shift
                model3_start_step = steps_end_second + steps_shift_third
                
                samples = KSamplerAdvanced().sample(
                    pipelines[2]['model'],
                    "disable",
                    noise_seed,
                    new_total_steps,
                    new_cfg,
                    sampler_name,
                    scheduler,
                    pipelines[2]['positive_conditioning'],
                    pipelines[2]['negative_conditioning'],
                    samples[0],
                    model3_start_step,
                    new_total_steps,
                    "enable"
                )
                print(f"Third model pass completed, latent shape: {samples[0]['samples'].shape}")
            except Exception as e:
                print(f"Error in third model pass: {str(e)}")
                raise e

        # Serialize the settings used for this run
        settings_str = self.serialize_settings(**effective_kwargs)
        
        # Return both the result and UI updates
        return {
            "ui": ui_updates,
            "result": (samples[0], settings_str),
        }