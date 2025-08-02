from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import gc

from datetime import datetime
from itertools import chain
from comfy.cli_args import args
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
from .adv_encode import advanced_encode

import comfy.sd
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_gits import GITSScheduler
import comfy_extras.nodes_model_advanced

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp"]

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
        founds = re.findall(self.tag_pattern, text)

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
        out = self._encode(clip, prompt, token_normalization, weight_interpretation)
        # encode and concatenate the rest of the prompt
        for prompt in prompts:
            cond_to = self._encode(clip, prompt, token_normalization, weight_interpretation)
            out = native.ConditioningConcat.concat(self, cond_to[0], out[0])
        return out


class MultiModelAdvancedKsampler:
    def __init__(self):
        self.loaded_checkpoints = {}  # Dictionary to store loaded checkpoints by name
        self.loaded_loras = {}        # Dictionary to store loaded LoRAs by path
        self.current_checkpoints = set()  # Track currently requested checkpoints
        
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
                "detached_seed": ("BOOLEAN", {"default": False}),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "load_settings": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("LATENT", "prompt", "seed", "lora", "settings")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    def print_memory_usage(self):
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print(f"Loaded checkpoints: {len(self.loaded_checkpoints)}")
        print(f"Loaded LoRAs: {len(self.loaded_loras)}")
        
    def cleanup(self):
        """Force cleanup of all resources"""
        print("Performing full cleanup of loaded models and LoRAs")
        
        # Clear checkpoints
        for name, (model, clip, vae) in self.loaded_checkpoints.items():
            del model
            del clip
            del vae
        self.loaded_checkpoints = {}
        
        # Clear LoRAs
        for path, lora in self.loaded_loras.items():
            del lora
        self.loaded_loras = {}
        
        self.current_checkpoints = set()
        
        # Force garbage collection
        gc.collect()
        comfy.model_management.soft_empty_cache()
        self.print_memory_usage()
        
    def serialize_settings(self, **kwargs):
        """Serialize all input parameters into a human-readable string"""
        used_model_count = kwargs.get("used_model_count", 2)
        
        settings = {
            "used_model_count": used_model_count,
            "ckpt_name1": kwargs.get("ckpt_name1", ""),
            **({"ckpt_name2": kwargs.get("ckpt_name2", "")} if used_model_count >= 2 else {}),
            **({"ckpt_name3": kwargs.get("ckpt_name3", "")} if used_model_count >= 3 else {}),
            "positive": kwargs.get("positive", ""),
            "negative": kwargs.get("negative", ""),
            "lora_name": kwargs.get("lora_name", ""),
            "noise_seed": kwargs.get("noise_seed", 0),
            "rescaled_steps": kwargs.get("rescaled_steps", 8),
            "rescale_multiplier": kwargs.get("rescale_multiplier", 0.7),
            "total_steps_original": kwargs.get("total_steps_original", 25),
            **({"total_steps_shift_second": kwargs.get("total_steps_shift_second", 0)} if used_model_count >= 2 else {}),
            **({"total_steps_shift_third": kwargs.get("total_steps_shift_third", 0)} if used_model_count >= 3 else {}),
            "sampler_name": kwargs.get("sampler_name", "euler"),
            "scheduler": kwargs.get("scheduler", "normal"),
            "starting_cfg": kwargs.get("starting_cfg", 8.0),
            "cfg_shift": kwargs.get("cfg_shift", 0.0),
            "steps_end_first": kwargs.get("steps_end_first", 15),
            **({"steps_shift_second": kwargs.get("steps_shift_second", 0)} if used_model_count >= 2 else {}),
            **({"steps_end_second": kwargs.get("steps_end_second", 0)} if used_model_count >= 2 else {}),
            **({"steps_shift_third": kwargs.get("steps_shift_third", 0)} if used_model_count >= 3 else {}),
            "token_normalization": kwargs.get("token_normalization", "none"),
            "weight_interpretation": kwargs.get("weight_interpretation", "comfy"),
            "detached_seed": kwargs.get("detached_seed", False),
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
        detached_seed = kwargs.get("detached_seed", False)
        
        if load_settings and load_settings.strip():
            loaded = self.deserialize_settings(load_settings)
            if loaded:
                # Create a new kwargs with loaded settings, but keep original latent_image
                effective_kwargs = loaded.copy()
                if "latent_image" in kwargs:
                    effective_kwargs["latent_image"] = kwargs["latent_image"]
                
                # If detached_seed is True, override the seed from direct input
                if detached_seed and "noise_seed" in kwargs:
                    effective_kwargs["noise_seed"] = kwargs["noise_seed"]
                    
                return effective_kwargs
        
        # Return None to indicate we should use direct inputs
        return None
    
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
            model, clip, vae = self.loaded_checkpoints[name]
            del model
            del clip
            del vae
            del self.loaded_checkpoints[name]
        
        # Update current checkpoints
        self.current_checkpoints = requested_set

    def load_model_pipeline(self, ckpt_name, lora_name, positive, negative,
                          token_normalization, weight_interpretation):
        """Helper function to load a complete model pipeline"""
        self.print_memory_usage()
        
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
            # Create a new LoraTagLoader instance for each call
            lora_loader = LoraTagLoader()
            model, clip, _ = lora_loader.load_lora(model, clip, lora_name)
            
            # Store the loaded LoRA reference if the loader has one
            if lora_loader.loaded_lora is not None:
                lora_path, lora = lora_loader.loaded_lora
                self.loaded_loras[lora_path] = lora
        
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
        try:
            # Print initial memory usage
            self.print_memory_usage()
            
            # Get effective settings (either from load_settings or direct inputs)
            effective_kwargs = self.get_effective_settings(**kwargs)
            
            if effective_kwargs is None:
                effective_kwargs = kwargs.copy()
            
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
                        "ckpt_name2": (loaded_settings.get("ckpt_name2", ""),),
                        "ckpt_name3": (loaded_settings.get("ckpt_name3", ""),),
                        "positive": (loaded_settings["positive"],),
                        "negative": (loaded_settings["negative"],),
                        "lora_name": (loaded_settings["lora_name"],),
                        "noise_seed": (loaded_settings["noise_seed"],),
                        "rescaled_steps": (loaded_settings["rescaled_steps"],),
                        "rescale_multiplier": (loaded_settings["rescale_multiplier"],),
                        "total_steps_original": (loaded_settings["total_steps_original"],),
                        "total_steps_shift_second": (loaded_settings.get("total_steps_shift_second", 0),),
                        "total_steps_shift_third": (loaded_settings.get("total_steps_shift_third", 0),),
                        "sampler_name": (loaded_settings["sampler_name"],),
                        "scheduler": (loaded_settings["scheduler"],),
                        "starting_cfg": (loaded_settings["starting_cfg"],),
                        "cfg_shift": (loaded_settings["cfg_shift"],),
                        "steps_end_first": (loaded_settings["steps_end_first"],),
                        "steps_shift_second": (loaded_settings.get("steps_shift_second", 0),),
                        "steps_end_second": (loaded_settings.get("steps_end_second", 0),),
                        "steps_shift_third": (loaded_settings.get("steps_shift_third", 0),),
                        "token_normalization": (loaded_settings["token_normalization"],),
                        "weight_interpretation": (loaded_settings["weight_interpretation"],),
                        "detached_seed": (loaded_settings.get("detached_seed", False),),
                    }

            # Extract all parameters from effective_kwargs
            used_model_count = effective_kwargs["used_model_count"]
            ckpt_name1 = effective_kwargs["ckpt_name1"]
            ckpt_name2 = effective_kwargs.get("ckpt_name2", "")
            ckpt_name3 = effective_kwargs.get("ckpt_name3", "")
            positive = effective_kwargs["positive"]
            negative = effective_kwargs["negative"]
            lora_name = effective_kwargs["lora_name"]
            noise_seed = effective_kwargs["noise_seed"]
            rescaled_steps = effective_kwargs["rescaled_steps"]
            rescale_multiplier = effective_kwargs["rescale_multiplier"]
            total_steps_original = effective_kwargs["total_steps_original"]
            total_steps_shift_second = effective_kwargs.get("total_steps_shift_second", 0)
            total_steps_shift_third = effective_kwargs.get("total_steps_shift_third", 0)
            sampler_name = effective_kwargs["sampler_name"]
            scheduler = effective_kwargs["scheduler"]
            starting_cfg = effective_kwargs["starting_cfg"]
            cfg_shift = effective_kwargs["cfg_shift"]
            steps_end_first = effective_kwargs["steps_end_first"]
            steps_shift_second = effective_kwargs.get("steps_shift_second", 0)
            steps_end_second = effective_kwargs.get("steps_end_second", 0)
            steps_shift_third = effective_kwargs.get("steps_shift_third", 0)
            token_normalization = effective_kwargs["token_normalization"]
            weight_interpretation = effective_kwargs["weight_interpretation"]
            latent_image = effective_kwargs["latent_image"]

            # Determine which checkpoints are being requested
            requested_checkpoints = [ckpt_name1]
            if used_model_count >= 2 and ckpt_name2:
                requested_checkpoints.append(ckpt_name2)
            if used_model_count >= 3 and ckpt_name3:
                requested_checkpoints.append(ckpt_name3)
            
            # Purge checkpoints that aren't being used this run
            self.purge_unused_checkpoints(requested_checkpoints)
            
            # Load all required pipelines
            pipelines = []
            try:
                pipelines.append(self.load_model_pipeline(ckpt_name1, lora_name, positive, negative,
                                                       token_normalization, weight_interpretation))
                if comfy.model_management.processing_interrupted():
                    raise RuntimeError("Processing interrupted by user")
                
                if used_model_count >= 2 and ckpt_name2:
                    pipelines.append(self.load_model_pipeline(ckpt_name2, lora_name, positive, negative,
                                                            token_normalization, weight_interpretation))
                    if comfy.model_management.processing_interrupted():
                        raise RuntimeError("Processing interrupted by user")
                
                if used_model_count >= 3 and ckpt_name3:
                    pipelines.append(self.load_model_pipeline(ckpt_name3, lora_name, positive, negative,
                                                            token_normalization, weight_interpretation))
                    if comfy.model_management.processing_interrupted():
                        raise RuntimeError("Processing interrupted by user")
            except Exception as e:
                self.cleanup()
                raise e

            samples = None
            try:
                # First sampling pass
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
                    raise RuntimeError("Processing interrupted by user")
                
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
                    raise RuntimeError("Processing interrupted by user")
                
                # Second model pass if enabled
                if used_model_count >= 2 and ckpt_name2:
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
                    raise RuntimeError("Processing interrupted by user")
                
                # Third model pass if enabled
                if used_model_count >= 3 and ckpt_name3:
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
                
                # Print final memory usage
                self.print_memory_usage()
                
                return {
                    "ui": ui_updates,
                    "result": (samples[0], positive, noise_seed, lora_name, settings_str),
                }
                
            except Exception as e:
                self.cleanup()
                if str(e) == "Processing interrupted by user":
                    print("Processing was interrupted, cleaned up resources")
                    return {
                        "ui": ui_updates,
                        "result": (latent_image, ""),
                    }
                else:
                    print(f"Error during sampling: {str(e)}")
                    raise e
            finally:
                if samples is None or comfy.model_management.processing_interrupted():
                    self.cleanup()
        except Exception as e:
            self.cleanup()
            raise e


class MultiModelPromptSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "settings": ("STRING", {"multiline": True}),
            },
            "optional": {
                "filename": (
                    "STRING",
                    {"default": "ComfyUI_%time_%seed_%counter", "multiline": False},
                ),
                "path": ("STRING", {"default": "%date/", "multiline": False}),
                "extension": (["png", "jpg", "jpeg", "webp"],),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "jpg_webp_quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "date_format": (
                    "STRING",
                    {"default": "%Y-%m-%d", "multiline": False},
                ),
                "time_format": (
                    "STRING",
                    {"default": "%H%M%S", "multiline": False},
                ),
                "save_metadata_file": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("FILENAME", "FILE_PATH", "METADATA")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "SD Prompt Reader"

    def save_images(
        self,
        images,
        settings: str,
        filename: str = "ComfyUI_%time_%seed_%counter",
        path: str = "%date/",
        extension: str = "png",
        lossless_webp: bool = True,
        jpg_webp_quality: int = 100,
        date_format: str = "%Y-%m-%d",
        time_format: str = "%H%M%S",
        save_metadata_file: bool = False,
        prompt=None,
        extra_pnginfo=None,
    ):
        (
            full_output_folder,
            filename_alt,
            counter_alt,
            subfolder_alt,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            self.prefix_append,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []
        files = []
        file_paths = []
        comments = []
        
        for image in images:
            variable_map = {
                "%date": self.get_time(date_format),
                "%time": self.get_time(time_format),
                "%extension": extension,
                "%quality": jpg_webp_quality,
            }

            subfolder = self.get_path(path, variable_map)
            output_folder = Path(full_output_folder) / subfolder
            output_folder.mkdir(parents=True, exist_ok=True)
            counter = self.get_counter(output_folder)
            variable_map["%counter"] = f"{counter:05}"

            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            stem = self.get_path(filename, variable_map)
            file = self.get_unique_filename(stem, extension, output_folder)
            file_path = output_folder / file

            if extension == "png":
                if not args.disable_metadata:
                    metadata = PngInfo()
                    metadata.add_text("parameters", settings)
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                img.save(
                    file_path,
                    pnginfo=metadata,
                    compress_level=4,
                )
            else:
                img.save(
                    file_path,
                    quality=jpg_webp_quality,
                    lossless=lossless_webp,
                )
                if not args.disable_metadata:
                    metadata = piexif.dump(
                        {
                            "Exif": {
                                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                    settings, encoding="unicode"
                                )
                            },
                        }
                    )
                    piexif.insert(metadata, str(file_path))

            if save_metadata_file:
                with open(file_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                    f.write(settings)

            results.append(
                {"filename": file.name, "subfolder": str(subfolder), "type": self.type}
            )
            files.append(str(file))
            file_paths.append(str(file_path))
            comments.append(settings)

        return {
            "ui": {"images": results},
            "result": (
                self.unpack_singleton(files),
                self.unpack_singleton(file_paths),
                self.unpack_singleton(comments),
            ),
        }

    @staticmethod
    def get_counter(directory: Path):
        img_files = list(
            chain(*(directory.rglob(f"*{suffix}") for suffix in SUPPORTED_FORMATS))
        )
        return len(img_files) + 1

    @staticmethod
    def get_path(name, variable_map):
        for variable, value in variable_map.items():
            name = name.replace(variable, str(value))
        return Path(name)

    @staticmethod
    def get_time(time_format):
        now = datetime.now()
        try:
            time_str = now.strftime(time_format)
            return time_str
        except:
            return ""

    @staticmethod
    def get_unique_filename(stem: Path, extension: str, output_folder: Path):
        file = stem.with_suffix(f"{stem.suffix}.{extension}")
        index = 0

        while (output_folder / file).exists():
            index += 1
            new_stem = Path(f"{stem}_{index}")
            file = new_stem.with_suffix(f"{new_stem.suffix}.{extension}")

        return file

    @staticmethod
    def unpack_singleton(arr: list):
        return arr[0] if len(arr) == 1 else arr