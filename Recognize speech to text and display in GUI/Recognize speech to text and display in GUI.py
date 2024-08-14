
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to speech
def speak_text(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Function to capture voice and display recognized text
def recognize_speech():
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            display_text.insert(tk.END, "Listening...\n")
            # Update the GUI
            root.update()
            # Listen to the user's input
            audio = r.listen(source, timeout=10)
            display_text.insert(tk.END, "Processing...\n")
            # Update the GUI
            root.update()
            # Recognize audio using Google
            text = r.recognize_google(audio)
            text = text.lower()
            display_text.insert(tk.END, "You said: " + text + "\n")
            # Update the GUI
            root.update()
            # Speak the recognized text
            speak_text(text)
    except sr.RequestError as e:
        display_text.insert(tk.END, "Could not request results; {0}\n".format(e))
    except sr.UnknownValueError:
        display_text.insert(tk.END, "Could not understand audio\n")
    # Update the GUI
    root.update()

# Initialize the GUI
root = tk.Tk()
root.title("Speech Recognition App")

# Create a text box to display the recognized text
display_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Arial", 12))
display_text.pack(padx=10, pady=10)

# Create a button to start the speech recognition
recognize_button = tk.Button(root, text="Recognize Speech", command=recognize_speech, font=("Arial", 14))
recognize_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
