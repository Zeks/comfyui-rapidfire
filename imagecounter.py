import os
from datetime import datetime

class ImmatureImageCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "date": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": "latent"})
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("file_count",)
    FUNCTION = "count_images"
    OUTPUT_NODE = True

    CATEGORY = "utils"

    def count_images(self, date, subfolder):
        dateFormat = "%Y-%m-%d"
        # Determine the correct date to use
        if not date:
            current_date = datetime.now().strftime(dateFormat)
        else:
            try:
                current_date = datetime.strptime(date, dateFormat).strftime(dateFormat)
            except ValueError:
                return (0,)

        # Define the image folder path
        image_folder_path = f"output/rapidfire/{current_date}/{subfolder}"

        # Initialize PNG file count
        png_count = 0

        # Count PNG files in the specified folder
        if os.path.exists(image_folder_path) and os.path.isdir(image_folder_path):
            for filename in os.listdir(image_folder_path):
                print("filename is:", filename )
                if filename.lower().endswith('.png'):
                    png_count += 1

        return (png_count,)