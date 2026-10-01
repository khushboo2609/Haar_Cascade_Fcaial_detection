import cv2
import os

DATASET_FOLDER = "dataset"

# Find first image
image_path = None

for root, folders, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(root, file)
            break

    if image_path:
        break

print("Testing image:")
print(image_path)

# Load image
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image could not be read.")
    exit()

print("Image size:", image.shape)

# Upscale
image = cv2.resize(
    image,
    None,
    fx=5,
    fy=5,
    interpolation=cv2.INTER_CUBIC
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Haar cascades to test
cascades = {
    "default": "haarcascade_frontalface_default.xml",
    "alt": "haarcascade_frontalface_alt.xml",
    "alt2": "haarcascade_frontalface_alt2.xml",
    "alt_tree": "haarcascade_frontalface_alt_tree.xml"
}

for name, filename in cascades.items():

    cascade_path = os.path.join(
        cv2.data.haarcascades,
        filename
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        print(name, "-> Cascade could not be loaded")
        continue

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(30, 30)
    )

    print(
        f"{name:10} -> {len(faces)} face(s) detected"
    )

    # Save result for each cascade
    result = image.copy()

    for (x, y, w, h) in faces:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    output_file = f"haar_{name}.jpg"

    cv2.imwrite(
        output_file,
        result
    )

    print("             Saved:", output_file)

print("\nHaar comparison completed.")