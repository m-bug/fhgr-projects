###
# dataset: https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
# download zip and extract it in the same folder as the script is. Name the folder "database".
# 
# Project structure must look like this:
# database/
# ├── Person A/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── Person B/
# │   ├── img1.jpg
# │   └── ...
# 
###

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

###
# Getting started:
# pip install opencv-python deepface matplotlib
###

# =================================================
# Globale Konfiguration
# =================================================
MODEL_NAME = "VGG-Face"
DETECTOR = "retinaface"


# =================================================
# Use Case 1: Face Verification (1:1)
# =================================================
def verify_faces(img1_path: str, img2_path: str):
    """
    Prüft, ob zwei Bilder dieselbe Person zeigen.
    """
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR
    )

    print("\n=== Use Case 1: Face Verification ===")
    print(f"Gleiche Person: {result['verified']}")
    print(f"Distanz: {result['distance']:.4f}")
    print(f"Threshold: {result['threshold']}")

    return result


# =================================================
# Use Case 2: Face Identification (1:N)
# =================================================
def identify_face(query_image: str, database_path: str):
    """
    Sucht ein Gesicht in einer lokalen Bilddatenbank.
    """
    print("\n=== Use Case 2: Face Identification ===")

    results = DeepFace.find(
        img_path=query_image,
        db_path=database_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True
    )

    if len(results) == 0 or results[0].empty:
        print("Keine Übereinstimmung gefunden.")
        return None

    df = results[0]
    best_match = df.iloc[0]

    print("Beste Übereinstimmung:")
    print(f"Bild: {best_match['identity']}")
    print(f"Distanz: {best_match['distance']:.4f}")

    return best_match


# =================================================
# Use Case 3: Emotion / Age Analysis
# =================================================
def analyze_emotion_age(image_path: str):
    """
    Analysiert Alter, Geschlecht und Emotion.
    """
    analysis = DeepFace.analyze(
        img_path=image_path,
        actions=["age", "gender", "emotion"],
        detector_backend=DETECTOR,
        enforce_detection=True
    )

    result = analysis[0]

    age = result["age"]
    gender = result["dominant_gender"]
    gender_conf = result["gender"][gender]
    emotion = result["dominant_emotion"]
    emotion_conf = result["emotion"][emotion]

    print("\n=== Use Case 3: Emotion & Age Analysis ===")
    print(f"Alter: {age}")
    print(f"Geschlecht: {gender} ({gender_conf:.1f} %)")
    print(f"Emotion: {emotion} ({emotion_conf:.1f} %)")

    display_image(
        image_path,
        title=f"Age: {age} | Gender: {gender} | Emotion: {emotion}"
    )

    return result


# =================================================
# Hilfsfunktion: Bild anzeigen
# =================================================
def display_image(image_path: str, title: str = ""):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title, fontsize=11)
    plt.show()


# =================================================
# Main
# =================================================
def main():
    # images from database
    img_angelina_1= "database/Angelina Jolie/001_fe3347c0.jpg"
    img_angelina_2= "database/Angelina Jolie/093_6ce62543.jpg"
    img_johnny_1= "database/Johnny Depp/005_9406f32d.jpg"
    img_cruise_1= "database/Tom Cruise/004_dc64d954.jpg"

    # images to test modell (not from database)
    img_test_angelina_jolie="testimages/angelina_jolie.jpeg"
    img_test_tom_hanks="testimages/tom_hanks.jpg"
    img_test_nicole_kidman="testimages/nicole_kidman.jpg"

    # database path
    database_path = "database"

    ###### START ######

    # 1a Verification (positive)
    verify_faces(img_angelina_1, img_angelina_2)

    # 1b Verification (negative) female with male
    verify_faces(img_angelina_1, img_johnny_1)

    # 1c Verification (negative) - make with male
    verify_faces(img_johnny_1, img_cruise_1)

    # 2. Identification
    # random images from google search
    identify_face(img_test_angelina_jolie, database_path)
    identify_face(img_test_tom_hanks, database_path)
    identify_face(img_test_nicole_kidman, database_path)

    # 3. Emotion & Age Analysis
    analyze_emotion_age(img_angelina_1)
    analyze_emotion_age(img_angelina_2)
    analyze_emotion_age(img_test_angelina_jolie)


if __name__ == "__main__":
    main()
