import os
import re
import hashlib

class BracketEscaper:
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
    FUNCTION = "escaped_brackets"
    OUTPUT_NODE = True

    CATEGORY = "utils"

    def escaped_brackets(self, string):
        string = string.strip()
        string = re.escape(string)
        string=string.replace("\(", "\\\\\\(")
        string=string.replace("\)", "\\\\\\)")
        return (string,)