import cv2
import os
import csv
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load model and label map
model = load_model("cricket_shot.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
rev_map = {v: k for k, v in label_map.items()}

# === Mediapipe Pose setup ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Predict shot
def predict_shot(frame):
    resized = cv2.resize(frame, (224, 224))
    img = resized / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    pred_index = np.argmax(prediction)
    pred_class = rev_map[pred_index]
    confidence = prediction[pred_index]
    return pred_class, confidence


# Setup paths 

output_base = "predictionFrames"
os.makedirs(output_base, exist_ok=True)

cap = cv2.VideoCapture(
    r"C:\Users\deepa\Cricket Shot Type Detection and Analysis\Notebook and Dataset\Short clip\drive.mp4"
)

# cap = cv2.VideoCapture(
#     r"C:\Users\deepa\Cricket Shot Type Detection and Analysis\Notebook and Dataset\Short clip\flick.mp4"
# )

# cap = cv2.VideoCapture(
#     r"C:\Users\deepa\Cricket Shot Type Detection and Analysis\Notebook and Dataset\Short clip\pull.mp4"
# )

# cap = cv2.VideoCapture(
#     r"C:\Users\deepa\Cricket Shot Type Detection and Analysis\Notebook and Dataset\Short clip\sweep.mp4"
# )

# cap = cv2.VideoCapture(
#     r"C:\Users\deepa\Cricket Shot Type Detection and Analysis\Notebook and Dataset\Short clip\istockphoto-1729389928-640_adpp_is.mp4"
# )


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    "final_output_video.mp4", fourcc, fps, (frame_width, frame_height)
)

csv_file = open("predictions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame_ID", "Predicted_Shot", "Confidence (%)", "Wrong_Shot"])

frame_id = 0

# Combine with Mediapipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

        # Predict shot
        pred_class, confidence = predict_shot(frame)
        wrong_shot = "Yes" if confidence < 0.8 else "No"

        #  Overlay label & prediction
        label_text = f"{pred_class.upper()} ({confidence*100:.2f}%)"
        cv2.putText(
            frame, label_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        if confidence < 0.8:
            cv2.putText(
                frame,
                "Wrong shot played",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        # Show frame, save video
        cv2.imshow("Cricket Shot Detection + Pose", frame)
        video_writer.write(frame)

        # Save frame image
        save_folder = os.path.join(output_base, pred_class)
        os.makedirs(save_folder, exist_ok=True)
        filename = f"frame_{frame_id}_{pred_class}_{int(confidence * 100)}.jpg"
        cv2.imwrite(os.path.join(save_folder, filename), frame)

        # Log CSV
        csv_writer.writerow([frame_id, pred_class, f"{confidence*100:.2f}", wrong_shot])
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#  Cleanup
cap.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"✅ Final video saved as final_output_video.mp4")
print(f"✅ CSV log saved as predictions.csv")
print(f"✅ {frame_id} frames processed and saved.")
