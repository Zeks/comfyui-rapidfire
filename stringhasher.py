import os
import hashlib

class StringHasher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": ""})
            }
        }
    
    @classmethod
    def IS_CHANGED(s):
        return float(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8)

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "pass_string"
    OUTPUT_NODE = True

    CATEGORY = "utils"

    def pass_string(self, string):
        return (string,)