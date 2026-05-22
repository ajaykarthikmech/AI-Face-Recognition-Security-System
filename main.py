from deepface import DeepFace
import cv2
import requests
import time
import threading
import numpy as np
import os

# ==============================
# TELEGRAM CONFIG
# ==============================

BOT_TOKEN = "YOUR_TOKEN"
CHAT_ID = "YOUR BOT ID"

face_status = "Scanning..."
last_sent = 0
ai_processing = False

# ==========================================
# LOAD KNOWN FACES
# ==========================================

KNOWN_FACES_DIR = "known_faces/allowed" 
#location of the files

known_embeddings = []

print("Loading known faces...")

for file in os.listdir(KNOWN_FACES_DIR):

    if file.endswith(".jpg") or file.endswith(".png"):

        path = os.path.join(KNOWN_FACES_DIR, file)

        name = os.path.splitext(file)[0]

        print(f"Loading {name}...")

        embedding = DeepFace.represent(
            img_path=path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]

        known_embeddings.append({
            "name": name,
            "embedding": np.array(embedding)
        })

print("All faces loaded.")

# ==========================================
# TELEGRAM FUNCTION
# ==========================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, data=payload)

# ==========================================
# FACE RECOGNITION FUNCTION
# ==========================================

def recognize_face(frame):

    global face_status
    global last_sent
    global ai_processing

    ai_processing = True

    try:

        # ==================================
        # CHECK FACE EXISTS
        # ==================================

        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )

        if len(faces) == 0:

            face_status = "No Face"

            ai_processing = False
            return

        # ==================================
        # GET CURRENT FACE EMBEDDING
        # ==================================

        embedding = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]

        embedding = np.array(embedding)

        # ==================================
        # FIND BEST MATCH
        # ==================================

        best_match = "Unknown"
        best_distance = 999999

        for person in known_embeddings:

            distance = np.linalg.norm(
                embedding - person["embedding"]
            )

            if distance < best_distance:

                best_distance = distance
                best_match = person["name"]

        # ==================================
        # MATCH THRESHOLD
        # ==================================

        if best_distance < 10:

            face_status = f"{best_match}"

            current_time = time.time()

            # ==================================
            # SEND TELEGRAM ALERT
            # ==================================

            if current_time - last_sent > 5:

                send_telegram(
                    f"{best_match} detected on webcam."
                )

                print(f"{best_match} alert sent.")

                last_sent = current_time

        else:

            face_status = "Unknown Person"

    except Exception as e:

        face_status = "AI Error"

        print("Error:", e)

    ai_processing = False

# ==========================================
# WEBCAM SETUP
# ==========================================

cap = cv2.VideoCapture(0)

# LOWER RESOLUTION = FASTER
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

last_ai_time = 0

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    current_time = time.time()

    # ======================================
    # RUN AI EVERY 0.5 SEC
    # ======================================

    if (
        current_time - last_ai_time > 0.5
        and not ai_processing
    ):

        threading.Thread(
            target=recognize_face,
            args=(frame.copy(),),
            daemon=True
        ).start()

        last_ai_time = current_time

    # ======================================
    # UI COLOR
    # ======================================

    color = (0, 255, 0)

    if "Unknown" in face_status:
        color = (0, 0, 255)

    elif "No Face" in face_status:
        color = (255, 255, 0)

    elif "Error" in face_status:
        color = (0, 255, 255)

    # ======================================
    # SHOW STATUS
    # ======================================

    cv2.putText(
        frame,
        face_status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.imshow(
        "AI Security Recognition System",
        frame
    )

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()