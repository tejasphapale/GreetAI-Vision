import cv2
import face_recognition
import os
import time
import pygame
import tempfile
import threading
from queue import Queue
from gtts import gTTS

# ================= AUDIO INIT =================
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
speech_queue = Queue()

def tts_worker():
    while True:
        text, lang = speech_queue.get()
        if text is None:
            break
        try:
            tts = gTTS(text=text, lang=lang, tld="co.in")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                path = f.name
                tts.save(path)

            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()
            os.remove(path)

        except Exception as e:
            print("[TTS ERROR]", repr(e))

        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text, lang="en"):
    speech_queue.put((text, lang))

# ================= WELCOME FUNCTIONS =================
def welcome_person(name):
    speak(f"नमस्कार {name}, आपल्या विभागात आपले स्वागत आहे", "mr")
    speak(f"Welcome to our department {name}", "en")

def welcome_guest():
    speak("नमस्कार, आपल्या विभागात आपले स्वागत आहे", "mr")
    speak("Welcome to our department", "en")


def special_welcome_yash():
    speak("""
यश..
शांत, संयमी आणि अभ्यासू वृत्ती असलेले यश.
आपल्या विभागात आपले हार्दिक स्वागत आहे.
""", "mr")
    speak("Welcome to our department, Yash", "en")

def special_welcome_sanjay_malpani_sir():
    speak("""
डॉक्टर. संजयजी मालपाणी..
परम तेजस्वी, विश्वगुरु, श्रीमद्भगवद्गीताचार्य, योगमहर्षी,
शिक्षण प्रसारक संस्थेचे कार्याध्यक्ष,
डॉक्टर. संजयजी मालपाणी सर...
जे काही करायचे ते अत्यंत भव्यदिव्य स्वरूपाचे असावे,
हा त्यांचा विचारच त्यांना विश्वगुरु बनवतो.
सदैव निरोगी, आनंदी आणि चिरतरुण राहण्यासाठी
योग साधना करायला हवी,
हा मौलिक विचार आपण आम्हाला देतात.
आजच्या युवा पिढीने समाजामध्ये वावरताना
सकारात्मक दृष्टिकोनाबरोबरच
स्वयंशिस्तीला अत्यंत महत्त्व द्यावे,
यासाठी संजुभाऊ वेळोवेळी आम्हाला मार्गदर्शन करतात.
सर्वगुणसंपन्न, प्रसन्न व्यक्तिमत्त्व,
सर्वांचे मार्गदर्शक,
योगमहर्षी,
डॉक्टर. संजय मालपाणी सर,
आपले सहर्ष स्वागत आहे.
""", "mr")

    speak(
        "A warm and respectful welcome to our department, Doctor Sanjay Malpani Sir.",
        "en"
    )

# ================= LOAD KNOWN FACES =================
print("[INFO] Loading known faces...")
known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    if file.lower().endswith((".jpg", ".png")):
        img = face_recognition.load_image_file(f"known_faces/{file}")
        enc = face_recognition.face_encodings(img)
        if enc:
            name = os.path.splitext(file)[0].lower()
            known_encodings.append(enc[0])
            known_names.append(name)
            print("[LOADED]", name)

print("[OK] Face database ready")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= MAIN LOOP =================
last_seen = {}
COOLDOWN = 20
frame_count = 0

print("\n🎉 AI FACE WELCOME SYSTEM STARTED")
print("Press Q to exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % 3 != 0:
        cv2.imshow("AI Welcome System", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break
        continue

    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    for enc, loc in zip(encodings, locations):
        name = "guest"
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
        if True in matches:
            name = known_names[matches.index(True)]

        now = time.time()
        if name not in last_seen or now - last_seen[name] > COOLDOWN:
            print("[DETECTED]", name)

            if name == "yash":
                special_welcome_yash()
            
            elif name == "sanjay malpani sir":
                special_welcome_sanjay_malpani_sir()
            elif name == "guest":
                welcome_guest()
            else:
                welcome_person(name)

            last_seen[name] = now

        top, right, bottom, left = [v * 4 for v in loc]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name.upper(), (left, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Welcome System", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

# ================= CLEAN EXIT =================
cap.release()
cv2.destroyAllWindows()
speech_queue.put((None, None))
pygame.mixer.quit()
print("✅ System closed safely")
