import comfy.sd
import folder_paths

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")

class CachedCheckpoint:
    def __init__(self):
        self.last_checkpoint = None
        self.last_checkpoint_path = None

    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }

    RETURN_TYPES = ("MODEL", any, "CLIP", "VAE")  # First output is truly any type
    RETURN_NAMES = ("MODEL", "Model name", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        if self.last_checkpoint_path == ckpt_path: 
            return self.last_checkpoint
        self.last_checkpoint_path = ckpt_path 
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        # Convert ckpt_name to a list (if it isn't already)
        ckpt_name_list = [ckpt_name] if isinstance(ckpt_name, str) else list(ckpt_name)
        
        # Insert list after the first element
        self.last_checkpoint = (out[0], *ckpt_name_list, *out[1:3])  # out[0], ckpt_name (as list elements), out[1], out[2]
        return self.last_checkpoint