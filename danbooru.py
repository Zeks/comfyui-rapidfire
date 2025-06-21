import datetime
import requests
import random
from typing import Literal
import os
import torch
import numpy as np
from PIL import Image
import io
import re

POST_AMOUNT = 100

def pil2tensor(image):
    """Convert a PIL image to a PyTorch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Booru():
    def __init__(self, booru, booru_url):
        self.booru = booru
        self.booru_url = booru_url
        self.headers = {'user-agent': 'my-app/0.0.1'}

    def get_data(self, add_tags, max_pages, id=''):
        pass

    def get_post(self, add_tags, max_pages, id=''):
        pass


class Danbooru(Booru):
    def __init__(self):
        super().__init__('danbooru', f'https://danbooru.donmai.us/posts.json?limit={POST_AMOUNT}')

    def get_data(self, add_tags, max_pages, id=''):
        if id:
            add_tags = ''

        for attempt in range(12):
            url = f"{self.booru_url}&page={random.randint(0, max_pages)}{add_tags}"
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                if data:
                    for post in data:
                        post['tags'] = post['tag_string']
                    return {'posts': data}
            except requests.RequestException as e:
                print(f"Request failed: {e}")

            max_pages = int(max_pages / 2)
            print(f"No data found, trying with page range: {max_pages}")

        return {'posts': []}

    def get_post(self, add_tags, max_pages, id=''):
        self.booru_url = f"https://danbooru.donmai.us/posts/{id}.json"
        try:
            response = requests.get(self.booru_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            data['tags'] = data['tag_string']
            return {'posts': [data]}
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return {'posts': []}


COLORED_BG = [
    'black_background', 'aqua_background', 'white_background',
    'colored_background', 'gray_background', 'blue_background',
    'green_background', 'red_background', 'brown_background',
    'purple_background', 'yellow_background', 'orange_background',
    'pink_background', 'plain', 'transparent_background',
    'simple_background', 'two-tone_background', 'grey_background'
]
BW_BG = ['monochrome', 'greyscale', 'grayscale']

RATING_TYPES = {
    "none": {"All": "All"},
    "full": {
        "All": "All", "Safe": "safe", "Sensitive": "questionable",
        "Questionable": "questionable", "Explicit": "explicit"
    },
    "single": {"All": "All", "Safe": "g", "Sensitive": "s", "Questionable": "q", "Explicit": "e"}
}
RATINGS = {"danbooru": RATING_TYPES['single'], }


class Ranbooru:
    def __init__(self):
        self.last_prompt = ''
        self.file_url = ''
        self.image = None
        self.last_rating = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_pages": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "number"}),
                "tags": ("STRING", {"multiline": False, "default": ""}),
                "replacement_tag": ("STRING", {"multiline": False, "default": ""}),  # New input
                "rating": (["All", "Safe", "Sensitive", "Questionable", "Explicit"], {"default": "All"}),
                "use_last_prompt": ("BOOLEAN", {"default": False}),
                "return_picture": ("BOOLEAN", {"default": False}),
                "alternate_ratings": ("BOOLEAN", {"default": False}),
                "blacklisted_tags": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    FUNCTION = "ranbooru"
    CATEGORY = "Ranbooru Nodes"

    def IS_CHANGED(self, **kwargs):
        return float('nan')

    def ranbooru(self, max_pages, tags, replacement_tag, rating, use_last_prompt, return_picture, alternate_ratings, blacklisted_tags):
        bad_tags = [
            'mixed-language_text', 'watermark', 'text', 'english_text', 'speech_bubble',
            'signature', 'artist_name', 'censored', 'bar_censor', 'translation',
            'twitter_username', "twitter_logo", 'patreon_username', 'commentary_request',
            'tagme', 'commentary', 'character_name', 'mosaic_censoring', 'instagram_username',
            'text_focus', 'english_commentary', 'comic', 'translation_request', 'fake_text',
            'translated', 'paid_reward_available', 'thought_bubble', 'multiple_views',
            'silent_comic', 'out-of-frame_censoring', 'symbol-only_commentary', '3koma', '2koma',
            'character_watermark', 'spoken_question_mark', 'japanese_text', 'spanish_text',
            'language_text', 'fanbox_username', 'commission', 'original', 'ai_generated',
            'stable_diffusion', 'tagme_(artist)', 'text_bubble', 'qr_code', 'chinese_commentary',
            'korean_text', 'partial_commentary', 'chinese_text', 'copyright_request', 'heart_censor',
            'censored_nipples', 'page_number', 'scan', 'fake_magazine_cover', 'korean_commentary'
        ]

        # Add blacklisted tags to the list of bad tags
        if blacklisted_tags:
            bad_tags.extend(blacklisted_tags.split(','))

        api_url = Danbooru()
        final_tags = ""
        if use_last_prompt and self.last_prompt != '':
            final_tags = self.last_prompt
            img_url = self.file_url
        else:
            add_tags = ''
            if tags:
                add_tags += f'&tags={tags.replace(",", "+")}'

            if rating != 'All':
                add_tags += f'+rating:{RATINGS["danbooru"][rating]}'

            data = api_url.get_data(add_tags, max_pages)

            random_post = self._select_random_post(data['posts'], alternate_ratings)

            clean_tags = re.sub(r'[()]', '', random_post['tags'])
            temp_tags = clean_tags.split()

            # Remove tags containing any blacklisted word
            filtered_tags = [tag for tag in temp_tags if not any(bad_word.strip() in tag for bad_word in bad_tags)]

            final_tags = ','.join(filtered_tags)
            self.last_prompt = final_tags

            if 'file_url' in random_post:
                self.file_url = random_post['file_url']
                print(f"Image URL: {self.file_url}")

            # Process replacement_tag logic
            if replacement_tag:
                # 1. Clean the search tags (remove parentheses and split)
                search_tags_cleaned = [re.sub(r'[()]', '', tag).strip() for tag in tags.split(',')]
                
                # 2. Split final_tags into a list for modification
                final_tags_list = [tag.strip() for tag in final_tags.split(',')]
                
                # 3. Find the first occurrence of any search tag and replace it
                for i, tag in enumerate(final_tags_list):
                    if tag in search_tags_cleaned:
                        final_tags_list[i] = replacement_tag
                        break  # Replace only the first occurrence (remove if all should be replaced)
                
                # 4. Join back into a string
                final_tags = ','.join(final_tags_list)
                self.last_prompt = final_tags

        if return_picture:
            img_url = self.file_url if use_last_prompt else random_post.get('file_url')
            try:
                response = requests.get(img_url)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                self.image = img
                return (final_tags, pil2tensor(img),)
            except Exception as e:
                print(f"Failed to fetch image: {e}")
                empty_image = Image.new('RGB', (1, 1), color=(0, 0, 0))
                return (final_tags, pil2tensor(empty_image),)
        else:
            empty_image = Image.new('RGB', (1, 1), color=(0, 0, 0))
            return (final_tags, pil2tensor(empty_image),)    

    def _select_random_post(self, posts, alternate_ratings):
        if not posts:
            return {'tags': 'bad_post_fix'}

        if not alternate_ratings or self.last_rating is None:
            random_post = random.choice(posts)
            self.last_rating = random_post.get('rating', None)
            return random_post

        opposite_rating = self._get_opposite_rating(self.last_rating)
        for post in posts:
            if post.get('rating') == opposite_rating:
                self.last_rating = opposite_rating
                return post

        # If no opposite rating found, just pick a random one
        random_post = random.choice(posts)
        self.last_rating = random_post.get('rating', None)
        return random_post

    def _get_opposite_rating(self, current_rating):
        ratings = list(RATINGS['danbooru'].values())
        index = ratings.index(current_rating)
        opposite_index = (index + 1) % len(ratings)
        return ratings[opposite_index]

NODE_CLASS_MAPPINGS = {
    "Ranbooru": Ranbooru,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ranbooru": "Ranbooru",
}