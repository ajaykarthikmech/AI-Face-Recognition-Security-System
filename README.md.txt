# AI-Based Real-Time Face Recognition Security System

## Introduction

This project is a real-time AI face recognition security system developed using Python, OpenCV, DeepFace, and Telegram Bot API integration.

The system detects and recognizes authorized persons through a webcam feed and sends instant Telegram alerts when a known person is detected.

---

## Features

- Real-time webcam face recognition
- Multi-person recognition
- Telegram alert integration
- Unknown-person filtering
- No-alert behavior for empty frames
- Optimized CPU performance
- Lightweight real-time architecture

---

## Technologies Used

- Python
- OpenCV
- DeepFace
- Facenet
- NumPy
- Telegram Bot API
- Threading

---

## Architecture Diagram

(Add your architecture diagram image here)

---

## Folder Structure

```text
FaceRecognitionProject/
│
├── main.py
├── requirements.txt
├── README.md
├── architecture_diagram.png
│
├── screenshots/
│      ├── detection.png
│      ├── telegram_alert.png
│
├── known_faces/
│      ├── allowed/
```

---

## Working Principle

1. Webcam captures live video feed.
2. Face detection checks whether a face exists.
3. Face embeddings are extracted using Facenet.
4. Embeddings are compared with authorized face database.
5. Telegram alert is generated for authorized persons only.

---

## Optimization Techniques

- Reduced webcam resolution
- Threaded AI inference
- Event-based alert system
- Preloaded embeddings
- Face existence checking

---

## Applications

- Smart surveillance
- Attendance systems
- Access control systems
- AI security prototypes
- Industrial monitoring systems

---

## Future Improvements

- GPU acceleration
- ESP32/Arduino integration
- Cloud database support
- YOLO integration
- Dashboard analytics
- Industrial AI inspection

---

## Author

Ajay Karthik