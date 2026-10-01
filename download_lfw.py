from sklearn.datasets import fetch_lfw_people
import os
import cv2
import numpy as np

DATASET_FOLDER = "dataset"

os.makedirs(DATASET_FOLDER, exist_ok=True)

print("Downloading LFW dataset...")

lfw_people = fetch_lfw_people(
    min_faces_per_person=5,
    resize=1.0,
    color=True
)

print("Download completed!")
print("Number of images:", len(lfw_people.images))
print("Number of people:", len(lfw_people.target_names))

for i, image in enumerate(lfw_people.images):

    # Get person's name
    person_name = lfw_people.target_names[
        lfw_people.target[i]
    ]

    # Create person's folder
    person_folder = os.path.join(
        DATASET_FOLDER,
        person_name
    )

    os.makedirs(person_folder, exist_ok=True)

    # IMPORTANT:
    # LFW images are floating point values approximately 0-1.
    # Convert them to 0-255 before saving.
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)

    # RGB -> BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR
    )

    filename = os.path.join(
        person_folder,
        f"{i+1}.jpg"
    )

    cv2.imwrite(filename, image_bgr)

print("\n================================")
print("LFW DATASET READY")
print("================================")
print("Location:", DATASET_FOLDER)