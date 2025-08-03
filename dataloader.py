import os
from datetime import datetime
import json

class DataExhaustedError(Exception):
    """Raised when loader has no more entries to return."""
    pass

class ImmatureImageDataLoader:
    def __init__(self):
        self._counter = 0
        self._loaded_seeds = []            # sorted list of seeds with images
        self._data_by_seed = {}           # metadata loaded from JSONL
        self._current_date = None
        self._subfolder = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Negative index resets internal counter; otherwise ignored.
                "index": ("INT", {"default": 0}),
                "date": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": "latent"})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("seed", "width", "height", "prompt")
    OUTPUT_NODE = True

    FUNCTION = "loadImageData"
    CATEGORY = "utils"

    def _load_if_needed(self, date, subfolder):
        date_format = "%Y-%m-%d"
        if not date:
            current_date = datetime.now().strftime(date_format)
        else:
            try:
                current_date = datetime.strptime(date, date_format).strftime(date_format)
            except ValueError:
                print(f"[loadImageData] invalid date '{date}', falling back to defaults")
                raise RuntimeError("Invalid date format")

        if self._current_date == current_date and self._subfolder == subfolder and self._loaded_seeds:
            return  # already loaded for this context

        # reset and reload
        self._current_date = current_date
        self._subfolder = subfolder
        self._counter = 0
        self._loaded_seeds = []
        self._data_by_seed = {}

        image_folder_path = os.path.join("output", "rapidfire", current_date, subfolder)
        jsonl_path = os.path.join("output", "rapidfire", current_date, "data.jsonl")

        image_filenames_dict = {}

        # Read image filenames
        if os.path.isdir(image_folder_path):
            for filename in os.listdir(image_folder_path):
                if filename.lower().endswith(".png"):
                    # Extract the seed part (before any underscore)
                    root = filename.split(".")[0]  # Remove extension
                    root = root.split("_")[0]     # Take part before first underscore
                    fullpath = os.path.join(image_folder_path, filename)
                    # Only keep the first occurrence if there are multiple suffixes
                    if root not in image_filenames_dict:
                        image_filenames_dict[root] = fullpath
        else:
            print(f"[loadImageData] image folder does not exist: {image_folder_path}")

        # Read JSONL data
        if os.path.isfile(jsonl_path):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            snippet = line[:80].replace("\n", "\\n")
                            print(f"[loadImageData] skipping malformed JSON line: {snippet}...")
                            continue
                        seed = entry.get("seed")
                        if seed is None:
                            continue
                        try:
                            seed = int(seed)
                        except (ValueError, TypeError):
                            continue

                        def safe_int(val, default):
                            try:
                                return int(val)
                            except (ValueError, TypeError):
                                return default

                        self._data_by_seed[seed] = {
                            "width": safe_int(entry.get("width"), 1024),
                            "height": safe_int(entry.get("height"), 1024),
                            "body": entry.get("body", "") or ""
                        }
            except Exception as e:
                print(f"[loadImageData] failed to read JSONL: {e}")
        else:
            print(f"[loadImageData] jsonl file does not exist: {jsonl_path}")

        # Attach image paths if available
        for filename_root, fullpath in image_filenames_dict.items():
            try:
                seed = int(filename_root)
            except ValueError:
                continue
            if seed in self._data_by_seed:
                self._data_by_seed[seed]["imagename"] = fullpath

        # Build sorted seed list based on available image filenames
        try:
            self._loaded_seeds = sorted(
                (int(k) for k in image_filenames_dict.keys() if k.isdigit()),
                reverse=False
            )
        except Exception:
            self._loaded_seeds = []

    def loadImageData(self, index, date, subfolder):
        # Negative index resets the internal counter
        if index < 0:
            self._counter = 0

        self._load_if_needed(date, subfolder)

        if self._counter >= len(self._loaded_seeds):
            raise DataExhaustedError("No more entries to process (exhausted all seeds)")

        seed = self._loaded_seeds[self._counter]
        entry = self._data_by_seed.get(seed, {})

        self._counter += 1

        return (
            seed,
            entry.get("width", 1024),
            entry.get("height", 1024),
            entry.get("body", "")
        )