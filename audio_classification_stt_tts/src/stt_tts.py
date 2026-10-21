"""Speech-to-Text and Text-to-Speech helper functions."""
def transcribe_audio_file(path):
    """Transcribe an audio file using Google Web Speech (internet required)."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(path)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech could not be understood."
    except sr.RequestError as exc:
        return f"STT service unavailable: {exc}"

def speak_text(text):
    """Speak text with pyttsx3 using an installed system voice."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(str(text))
    engine.runAndWait()
    engine.stop()
