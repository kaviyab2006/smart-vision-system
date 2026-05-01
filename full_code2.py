import threading
import speech_recognition as sr
import pyttsx3
import time

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Global variable for specific object
specific_object = None

# List of allowed objects
ALLOWED_OBJECTS = ['person', 'chair', 'laptop', 'cell phone', 'book',
                   'handbag', 'backpack', 'bottle']

# Function to speak
def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# Voice command listener
def listen_commands():
    global specific_object
    r = sr.Recognizer()
    
    # Pick the correct microphone index here
    mic = sr.Microphone(device_index=2)
    
    while True:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                print("🎙️ Listening for voice command...")
                audio = r.listen(source, phrase_time_limit=3)
                command = r.recognize_google(audio).lower()
                print("🗣️ You said:", command)

                if "find" in command:
                    for obj in ALLOWED_OBJECTS:
                        if obj in command:
                            specific_object = obj
                            speak(f"Tracking {obj} now")
                            print(f"🎤 Command received: Track {obj}")
                            break
                elif "stop tracking" in command:
                    specific_object = None
                    speak("Tracking all objects now")
                    print("🎤 Command received: Stop tracking")
        except sr.UnknownValueError:
            pass  # ignore unrecognized speech
        except Exception as e:
            print("❌ Voice listener error:", e)

# Start voice listener in background thread
threading.Thread(target=listen_commands, daemon=True).start()

# Main loop placeholder
while True:
    # Here you would run your YOLO detection and use 'specific_object' to filter
    # For testing, just show the current tracking target:
    if specific_object:
        print(f"Currently tracking: {specific_object}")
    else:
        print("Tracking all objects")
    time.sleep(1)
