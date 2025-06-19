import os
import random
import time
from typing import Dict, List

class CharacterDatabase:
    def __init__(self):
        self.hardcoded_path = r"H:\imagegen checkpoints\loras\kiseki_illustrious_v2\prompts\kiseki-noclothes-illustrious.txt"
        self.characters: Dict[str, dict] = {}
        self.character_names: List[str] = []
        self.load_data()

    def load_data(self):
        """Load and parse the character data file"""
        self.characters = {}
        self.character_names = []
        
        if not os.path.exists(self.hardcoded_path):
            raise FileNotFoundError(f"File not found at: {self.hardcoded_path}")

        with open(self.hardcoded_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',', 2)
                if len(parts) < 3:
                    continue
                
                lora_part = parts[0].strip()
                name_part = parts[1].strip()
                tags_part = parts[2].strip()
                
                self.characters[name_part] = {
                    'lora': lora_part,
                    'tags': tags_part
                }
                self.character_names.append(name_part)

class RandomCharacterSelector:
    @classmethod
    def INPUT_TYPES(cls):
        default_names = []
        try:
            db = CharacterDatabase()
            default_names = db.character_names
        except Exception as e:
            print(f"Note: Could not pre-load character names: {e}")

        return {
            "required": {
                "num_girls": ("INT", {"default": 1, "min": 1, "max": 4}),
                "girl1_mode": (["random", "specified", "disabled"], {"default": "random"}),
                "girl1_name": (default_names, {"default": ""}),
                "girl1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "girl2_mode": (["random", "specified", "disabled"], {"default": "disabled"}),
                "girl2_name": (default_names, {"default": ""}),
                "girl2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "girl3_mode": (["random", "specified", "disabled"], {"default": "disabled"}),
                "girl3_name": (default_names, {"default": ""}),
                "girl3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "girl4_mode": (["random", "specified", "disabled"], {"default": "disabled"}),
                "girl4_name": (default_names, {"default": ""}),
                "girl4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "full_prompt", 
        "name1", "name2", "name3", "name4",
        "tags1", "tags2", "tags3", "tags4",
        "lora1", "lora2", "lora3", "lora4"
    )
    FUNCTION = "process"
    CATEGORY = "conditioning"

    def __init__(self):
        self.database = CharacterDatabase()
        self.last_run_time = 0

    def process(self, num_girls, **kwargs):
        random.seed(int(time.time() * 1000))
        
        results = {
            'full_prompt': '',
            'names': ['', '', '', ''],
            'tags': ['', '', '', ''],
            'loras': ['', '', '', '']
        }

        for i in range(1, 5):
            if i > num_girls:
                continue

            mode = kwargs.get(f'girl{i}_mode', 'disabled')
            if mode == 'disabled':
                continue

            name = kwargs.get(f'girl{i}_name', '')
            strength = kwargs.get(f'girl{i}_strength', 1.0)
            
            if mode == 'random':
                if not self.database.character_names:
                    continue
                name = random.choice(self.database.character_names)
            elif mode == 'specified':
                if not name or name not in self.database.characters:
                    continue

            character = self.database.characters[name]
            
            # Process LoRA
            lora_part = character['lora']
            if ':' in lora_part:
                lora_parts = lora_part.split(':')
                if len(lora_parts) >= 2:
                    lora_name = lora_parts[1].replace('>', '')
                    processed_lora = f"<lora:{lora_name}:{strength}>"
                    results['loras'][i-1] = processed_lora
            
            # Create prompt entry without LoRA
            entry = f"{name}, {character['tags']}"
            
            if results['full_prompt']:
                results['full_prompt'] += ", "
            results['full_prompt'] += "(" + entry + ":1.2)"
            
            results['names'][i-1] = name
            results['tags'][i-1] = character['tags']

        self.last_run_time = time.time()

        return (
            results['full_prompt'],
            *results['names'],
            *results['tags'],
            *results['loras']
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

NODE_CLASS_MAPPINGS = {
    "RandomCharacterSelector": RandomCharacterSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomCharacterSelector": "Character Prompt Generator"
}