<<<<<<< HEAD
# 🏏 Cricket Shot Type Detection and Analysis

## 📌 Overview

This project aims to detect and classify cricket shot types (e.g., Drive, Pull, Flick, Sweep) using a deep learning model combined with pose estimation. It provides real-time predictions with confidence scores, overlays the result on the video, and logs each prediction for later analysis.

---

## 🎯 Purpose

- To automate cricket shot analysis for coaching and sports research.
- To combine computer vision and deep learning in real-time video applications.
- To generate insights from body posture and shot prediction accuracy.
- To save and organize visual data frame-wise for performance review.

---

## 🔍 What the Project Does

- Loads a pre-trained CNN model to classify shots.
- Uses Mediapipe to detect player pose in each video frame.
- Predicts the type of shot and its confidence level.
- Flags shots with low confidence as possibly incorrect ("Wrong shot played").
- Saves annotated video frames organized by shot class.
- Generates a full output video with pose and prediction overlays.
- Creates a CSV log (`predictions.csv`) with frame-wise predictions.

---

## ⚙️ How it Works

1. **Model Loading**  
   Loads the `cricket_shot.h5` model and `label_map.pkl` to decode predicted classes.

2. **Pose Detection**  
   Mediapipe tracks the player's pose and draws it on the video frame.

3. **Shot Classification**  
   Each frame is resized, normalized, and passed through the model for prediction.

4. **Output Generation**  
   - Predicted label and confidence are overlayed.
   - "Wrong shot played" is displayed if confidence < 50%.
   - Frames are saved in `ShotFrames/<Class>` folders.
   - The full output video is saved as `final_output_video.mp4`.
   - A detailed log is saved in `predictions.csv`.

---

## 🛠️ Technologies Used

| Category         | Tools & Libraries              |
|------------------|-------------------------------|
| Language         | Python                        |
| Deep Learning    | TensorFlow / Keras            |
| Computer Vision  | OpenCV                        |
| Pose Detection   | Mediapipe                     |
| Data Handling    | NumPy, CSV, Pickle            |
| Model            | CNN (Image Classification)    |
| Output Files     | MP4 Video, CSV Log, Image Frames |

---

## 📁 Output Files

- `final_output_video.mp4` – Annotated full video
- `predictions.csv` – Frame-wise prediction and confidence log
- `ShotFrames/` – Saved annotated frames per predicted shot class

---

## 💡 Key Features

- Combines **pose detection** with **deep learning-based classification**.
- Detects and labels shots with confidence.
- Flags uncertain predictions (confidence < 50%) as potentially incorrect.
- Organizes output frames and logs systematically.
- Useful for coaching, training analytics, and research in cricket.

---

## 🧠 Why This Project?

This project bridges the gap between AI and sports. While many datasets exist for object recognition, cricket-specific activity recognition is niche and growing. This work demonstrates:

- Real-time video inference using CNNs.
- Pose-guided activity detection.
- Sports-specific AI applications.

---

## 🚀 Future Scope

- Expand to include more shot classes.
- Use LSTM for video-based temporal learning.
- Implement player tracking and feedback modules.
- Create a web-based interface for coach review.

---

## 📢 Author Note

If you're a sports analyst, AI enthusiast, or cricket fan — this project is for you! It's a small step towards making intelligent, real-time sports analytics accessible to all.

---

## 📬 Contact

Feel free to reach out for collaboration, suggestions, or implementation help!
=======
# cricket-shot-type-detection-analysis-
>>>>>>> 163320c774be955e351706cf4341d2d5bc9e9bda
