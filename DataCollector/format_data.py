import os
from PIL import Image

# Define the paths to the source and target directories
source_dir = 'data'
left_dir = 'left'
right_dir = 'right'
both_dir = 'joined'

# Make the target directories if they don't exist
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)
os.makedirs(both_dir, exist_ok=True)

# Iterate over the images in the source directory
for i, filename in enumerate(os.listdir(source_dir)):
    # Open the image
    image = Image.open(os.path.join(source_dir, filename))
    width, height = image.size
    # Split the image in half vertically
    left_image = image.crop((0, 0, width//2, height))
    right_image = image.crop((width//2, 0, width, height))
    # Save the left and right halves in their respective directories
    left_image.save(os.path.join(left_dir, f"left{i}.jpg"))
    right_image.save(os.path.join(right_dir, f"right{i}.jpg"))
    left_image.save(os.path.join(both_dir, f"left{i}.jpg"))
    right_image.save(os.path.join(both_dir, f"right{i}.jpg"))
    # Save both halves together in the 'both' directory
    #both_image = Image.new('RGB', (width, height))
    #both_image.paste(left_image, (0, 0))
    #both_image.paste(right_image, (width//2, 0))
    #both_image.save(os.path.join(both_dir, f"both{i}.jpg"))
