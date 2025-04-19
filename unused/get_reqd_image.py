import os
from PIL import Image
import matplotlib.pyplot as plt

def get_image(image_id):
    filename = f"{image_id}.jpg"
    path1 = os.path.join("./VisualGenome/VG_100K/", filename)
    path2 = os.path.join("./VisualGenome/VG_100K_2/", filename)

    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        raise FileNotFoundError(f"Image {filename} not found in either folder.")

# Example usage
image_id = "2406818"  # Replace with actual ID
try:
    image_path = get_image(image_id)
    img = Image.open(image_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image ID: {image_id}")
    output_path = f"{image_id}_preview.png"
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

except FileNotFoundError as e:
    print(e)
