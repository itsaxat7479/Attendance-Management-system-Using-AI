import os
import pickle
import face_recognition
import cv2

# Paths
dataset_dir = "dataset"
model_file = "face_encodings.pkl"

def train_model():
    """Train the face recognition model by encoding images in the dataset."""
    print("Training the model...")

    known_encodings = []
    known_names = []

    # Process each user folder in the dataset
    for folder in os.listdir(dataset_dir):
        user_folder = os.path.join(dataset_dir, folder)
        if os.path.isdir(user_folder):
            for image_file in os.listdir(user_folder):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(user_folder, image_file)
                    image = cv2.imread(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(folder)

    # Save encodings to a file
    with open(model_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"Model training complete. Encodings saved to '{model_file}'.")


if __name__ == "__main__":
    train_model()
