import cv2
import os
import time

# ==========================================
# PATHS
# ==========================================

DATASET_FOLDER = "dataset"
OUTPUT_FOLDER = "Output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# LOAD HAAR CASCADE
# ==========================================

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Haar Cascade could not be loaded.")
    exit()

print("Haar Cascade loaded successfully.")


# ==========================================
# SUPPORTED IMAGE FORMATS
# ==========================================

supported_formats = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ==========================================
# STATISTICS
# ==========================================

total_images = 0
images_with_faces = 0
images_without_faces = 0
total_faces = 0

start_time = time.time()


# ==========================================
# PROCESS DATASET
# ==========================================

for root, folders, files in os.walk(DATASET_FOLDER):

    for filename in files:

        if not filename.lower().endswith(supported_formats):
            continue

        input_path = os.path.join(
            root,
            filename
        )

        # Read image
        image = cv2.imread(input_path)

        if image is None:
            print("Could not read:", input_path)
            continue

        total_images += 1

        # ==================================
        # UPSCALE SMALL LFW IMAGE
        # ==================================

        image = cv2.resize(
            image,
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        # ==================================
        # GRAYSCALE
        # ==================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================
        # CONTRAST ENHANCEMENT
        # ==================================

        gray = cv2.equalizeHist(gray)

        # ==================================
        # FACE DETECTION
        # ==================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(30, 30)
        )

        number_of_faces = len(faces)

        total_faces += number_of_faces

        # ==================================
        # STATISTICS
        # ==================================

        if number_of_faces > 0:
            images_with_faces += 1
        else:
            images_without_faces += 1

        # ==================================
        # GET PERSON NAME
        # ==================================

        person_name = os.path.basename(root)

        print(
            f"{person_name} / {filename} -> "
            f"{number_of_faces} face(s) detected"
        )

        # ==================================
        # DRAW BOUNDING BOXES
        # ==================================

        for (x, y, w, h) in faces:

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ==================================
        # PRESERVE PERSON FOLDER
        # ==================================

        relative_folder = os.path.relpath(
            root,
            DATASET_FOLDER
        )

        output_person_folder = os.path.join(
            OUTPUT_FOLDER,
            relative_folder
        )

        os.makedirs(
            output_person_folder,
            exist_ok=True
        )

        # ==================================
        # SAVE RESULT
        # ==================================

        output_path = os.path.join(
            output_person_folder,
            filename
        )

        cv2.imwrite(
            output_path,
            image
        )


# ==========================================
# FINAL STATISTICS
# ==========================================

end_time = time.time()

processing_time = end_time - start_time

if total_images > 0:
    detection_percentage = (
        images_with_faces / total_images
    ) * 100
else:
    detection_percentage = 0


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("       HAAR CASCADE RESULTS")
print("======================================")

print(
    "Total images processed:",
    total_images
)

print(
    "Images with detected faces:",
    images_with_faces
)

print(
    "Images without detected faces:",
    images_without_faces
)

print(
    "Total faces detected:",
    total_faces
)

print(
    f"Detection rate: {detection_percentage:.2f}%"
)

print(
    f"Processing time: {processing_time:.2f} seconds"
)

print(
    f"Average time per image: "
    f"{processing_time / total_images:.4f} seconds"
    if total_images > 0
    else "Average time per image: N/A"
)

print(
    "Results saved in:",
    OUTPUT_FOLDER
)

print("======================================")