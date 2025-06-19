import os
import random
import time
from typing import Dict, List

class CharacterDatabase:
    def __init__(self):
        self.hardcoded_path = r"H:\imagegen checkpoints\loras\kiseki_illustrious_v2\prompts\kiseki-noclothes-illustrious_weighted.txt"
        self.characters: Dict[str, dict] = {}
        self.character_names: List[str] = []
        self.character_weights: List[int] = []
        self.load_data()

    def load_data(self):
        """Load and parse the character data file with weights"""
        self.characters = {}
        self.character_names = []
        self.character_weights = []
        
        if not os.path.exists(self.hardcoded_path):
            raise FileNotFoundError(f"File not found at: {self.hardcoded_path}")

        with open(self.hardcoded_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',', 3)  # Split into weight, lora, name, tags
                if len(parts) < 4:
                    continue
                
                try:
                    weight = int(parts[0].strip())
                except ValueError:
                    weight = 1  # Default weight if parsing fails
                
                lora_part = parts[1].strip()
                name_part = parts[2].strip()
                tags_part = parts[3].strip()
                
                self.characters[name_part] = {
                    'weight': weight,
                    'lora': lora_part,
                    'tags': tags_part
                }
                self.character_names.append(name_part)
                self.character_weights.append(weight)

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
                "num_girls": ("INT", {"default": 1, "min": 1, "max": 2}),
                "girl1_mode": (["random", "specified", "disabled"], {"default": "random"}),
                "girl1_name": (default_names, {"default": ""}),
                "girl1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "girl1_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "girl2_mode": (["random", "specified", "disabled"], {"default": "disabled"}),
                "girl2_name": (default_names, {"default": ""}),
                "girl2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "girl2_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "full_prompt", 
        "name1", "name2",
        "tags1", "tags2",
        "lora1", "lora2"
    )
    FUNCTION = "process"
    CATEGORY = "conditioning"

    def __init__(self):
        self.database = CharacterDatabase()
        self.last_run_time = 0

    def weighted_random_choice(self):
        """Select a random character based on weights"""
        return random.choices(
            self.database.character_names,
            weights=self.database.character_weights,
            k=1
        )[0]

    def process(self, num_girls, **kwargs):
        random.seed(int(time.time() * 1000))
        
        results = {
            'full_prompt': '',
            'names': ['', ''],
            'tags': ['', ''],
            'loras': ['', '']
        }

        enabled_girls = 0
        # First pass to count enabled girls
        for i in range(1, 3):
            if i > num_girls:
                continue
            mode = kwargs.get(f'girl{i}_mode', 'disabled')
            if mode != 'disabled':
                enabled_girls += 1

        # Add girl count prefix
        if enabled_girls > 0:
            results['full_prompt'] = f"{enabled_girls}girl{'s' if enabled_girls > 1 else ''}, "

        # Second pass to process girls
        for i in range(1, 3):
            if i > num_girls:
                continue

            mode = kwargs.get(f'girl{i}_mode', 'disabled')
            if mode == 'disabled':
                continue

            name = kwargs.get(f'girl{i}_name', '')
            strength = kwargs.get(f'girl{i}_strength', 1.0)
            lora_strength = kwargs.get(f'girl{i}_lora_strength', 1.0)
            
            if mode == 'random':
                if not self.database.character_names:
                    continue
                name = self.weighted_random_choice()
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
                    processed_lora = f"<lora:{lora_name}:{lora_strength}>"
                    results['loras'][i-1] = processed_lora
            
            # Create prompt entry with strength if not 1.0
            processed_name = f"({name}:{strength})" if strength != 1.0 else name
            entry = f"{processed_name}, {character['tags']}"
            
            if len(results['full_prompt']) > len(f"{enabled_girls}girl{'s' if enabled_girls > 1 else ''}, "):
                results['full_prompt'] += ", "
            results['full_prompt'] += entry
            
            results['names'][i-1] = name
            results['tags'][i-1] = character['tags']

        self.last_run_time = time.time()

        return (
            results['full_prompt'],
            results['names'][0],
            results['names'][1],
            results['tags'][0],
            results['tags'][1],
            results['loras'][0],
            results['loras'][1]
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