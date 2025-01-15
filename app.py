import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import numpy as np

# Load the YOLO model
model_path = 'model/v11_170ep.pt'
model = YOLO(model_path)

# Paths
dataset_path = 'dataset'  # Folder for input images
output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)

# Class names
class_names = [
    "Verdes", "Pintoness", "Maduros", "Sobremaduros", "Secos"
]

# Colors for classes
class_colors = [
    (0, 255, 0, 100),  # Green for Class 1
    (255, 255, 0, 100),  # Yellow for Class 3
    (255, 0, 0, 100),  # Red for Class 0
    (255, 255, 255, 100),  # Black for Class 2
    (239, 239, 240, 100)  # Gray for Class 4
]

def predict_and_draw_segmentation(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')

    # YOLO model predicts directly from the image path
    results = model.predict(image, save=False, save_txt=False)

    # Load image for drawing
    image_with_mask = image.copy()
    mask_layer = Image.new("RGBA", image_with_mask.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    # Initialize counters
    class_counts = {}

    # Process segmentation results
    for result in results:
        for seg, cls in zip(result.masks.data, result.boxes.cls):
            cls = int(cls)  # Convert class index to integer
            color = class_colors[cls % len(class_colors)]  # Get color for class

            # Convert segmentation mask to a polygon
            seg_np = seg.cpu().numpy().astype(np.uint8)  # Convert to NumPy array
            contours, _ = cv2.findContours(seg_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Scale contour coordinates to match image size
                contour = contour.reshape(-1, 2) * np.array(
                    [image.size[0] / seg_np.shape[1], image.size[1] / seg_np.shape[0]]
                )
                contour = contour.astype(np.int32)

                # Convert contour to list of tuples
                contour = [tuple(point) for point in contour]

                # Draw outline for better visibility
                draw.line(contour + [contour[0]], fill=(0, 0, 0, 255), width=2)

            # Count the class
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += 1

    # Merge mask with the original image
    image_with_mask = Image.alpha_composite(image_with_mask.convert("RGBA"), mask_layer)

    # Add summary text
    total_count = sum(class_counts.values())
    percentages = {cls: (count / total_count) * 100 for cls, count in class_counts.items()}
    summary_text = [f"{class_names[cls]}: {count} ({percent:.2f}%)" for cls, (count, percent) in zip(class_counts.keys(), percentages.items())]
    summary_text.insert(0, f"Total: {total_count}")

    # Create new image with space for the summary text
    text_height = (len(summary_text) + 1) * 20 + 10  # Calculate space needed for the text
    new_height = image_with_mask.size[1] + text_height
    extended_image = Image.new("RGBA", (image_with_mask.size[0], new_height), (0, 0, 0, 255))
    extended_image.paste(image_with_mask, (0, 0))  # Paste the original image on top

    # Add summary text in the extended area
    draw_summary = ImageDraw.Draw(extended_image)
    font = ImageFont.load_default()
    text_x, text_y = 10, image_with_mask.size[1] + 10  # Start below the original image

    for line in summary_text:
        draw_summary.text((text_x, text_y), line, fill="white", font=font)
        text_y += 20

    # Save result as PNG (to preserve transparency)
    result_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")
    extended_image.save(result_path)

# Process all images
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(root, file)
            predict_and_draw_segmentation(image_path)

print("Segmentation complete.")
