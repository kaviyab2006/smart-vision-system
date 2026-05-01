""""import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    print("🎙️ Say something...")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    print("Processing...")
    try:
        text = r.recognize_google(audio)
        print("✅ You said:", text)
    except Exception as e:
        print("❌ Error:", e)"""

import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=0)

with mic as source:
    print("🎤 Speak now...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio)
    print("You said:", text)

