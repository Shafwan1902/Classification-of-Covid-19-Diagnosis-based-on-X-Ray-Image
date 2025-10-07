from PIL import Image
import os

# Folder containing normal class images
folder_path = "C:\\Users\\user\\Desktop\\deeplearning\\dataset\\normal"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(folder_path, filename)
        jpg_filename = filename.replace(".png", ".jpg")
        jpg_path = os.path.join(folder_path, jpg_filename)

        # Convert and save
        with Image.open(png_path).convert("RGB") as img:
            img.save(jpg_path, "JPEG")
        
        # Optionally delete original .png file
        os.remove(png_path)
        print(f"Converted and deleted: {filename}")
