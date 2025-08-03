import os
import json
from datetime import datetime

class CsvWriterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 23948457612}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "body": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "append_to_jsonl"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    def append_to_jsonl(self, seed, width, height, body):
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join("output", "rapidfire", current_date)
        os.makedirs(output_dir, exist_ok=True)
        jsonl_path = os.path.join(output_dir, "data.jsonl")

        # Compose the record
        record = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "seed": int(seed),
            "width": int(width) if isinstance(width, (int, str)) and str(width).isdigit() else 1024,
            "height": int(height) if isinstance(height, (int, str)) and str(height).isdigit() else 1024,
            "body": body or ""
        }

        # Append as a single JSON line
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[JsonAppenderNode] failed to write JSONL: {e}")

        return ()
