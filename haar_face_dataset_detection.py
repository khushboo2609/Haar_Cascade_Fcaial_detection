import cv2
import os

# ==========================================
# 1. FOLDER PATHS
# ==========================================

DATASET_FOLDER = "dataset"
OUTPUT_FOLDER = "Output"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# 2. LOAD HAAR CASCADE
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Check whether Haar Cascade loaded correctly
if face_cascade.empty():
    print("ERROR: Haar Cascade could not be loaded.")
    exit()

print("Haar Cascade loaded successfully.")


# ==========================================
# 3. SUPPORTED IMAGE FORMATS
# ==========================================

supported_formats = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ==========================================
# 4. READ LFW DATASET
# ==========================================

total_images = 0
total_faces = 0
images_with_faces = 0
images_without_faces = 0


# os.walk() goes through all person folders
for root, folders, files in os.walk(DATASET_FOLDER):

    for filename in files:

        # Check image format
        if not filename.lower().endswith(supported_formats):
            continue

        # Complete input path
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


        # ======================================
        # 5. CONVERT IMAGE TO GRAYSCALE
        # ======================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # ======================================
        # 6. DETECT FACES
        # ======================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )


        # Count detected faces
        number_of_faces = len(faces)

        total_faces += number_of_faces


        # ======================================
        # 7. COUNT DETECTION RESULTS
        # ======================================

        if number_of_faces > 0:
            images_with_faces += 1
        else:
            images_without_faces += 1


        # Get person name from folder
        person_name = os.path.basename(root)


        print(
            f"{person_name} / {filename} -> "
            f"{number_of_faces} face(s) detected"
        )


        # ======================================
        # 8. DRAW FACE BOUNDING BOXES
        # ======================================

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


        # ======================================
        # 9. CREATE SAME PERSON FOLDER IN OUTPUT
        # ======================================

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


        # ======================================
        # 10. SAVE RESULT
        # ======================================

        output_path = os.path.join(
            output_person_folder,
            filename
        )

        cv2.imwrite(
            output_path,
            image
        )


# ==========================================
# 11. FINAL RESULTS
# ==========================================

print("\n================================")
print("LFW HAAR PROCESSING COMPLETED")
print("================================")

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
    "Results saved in:",
    OUTPUT_FOLDER
)