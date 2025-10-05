import speech_recognition as sr
import threading
from queue import Queue
import wikipedia

wikipedia.set_lang("ar")  # default language

# Detect speech language (Arabic or English)
def detect_language(text):
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
    return "ar" if arabic_chars >= latin_chars else "en"

class PersonSpeech:
    def __init__(self, face_id):
        self.face_id = face_id  # link person with Face ID
        self.history = []
        self.queue = Queue()
        self.reply_queue = Queue()
        self.lock = threading.Lock()
        self.listening = False
        self.recognizer = sr.Recognizer()
        self.thread = None
        self.lang = None

    def generate_reply(self, text):
        try:
            wikipedia.set_lang(self.lang or "ar")
            summary = wikipedia.summary(text, sentences=2)
            return summary
        except wikipedia.DisambiguationError as e:
            return f"Ambiguous question: {e.options[:5]}"
        except wikipedia.PageError:
            return "Sorry, no information found."
        except Exception as e:
            print(f"Wikipedia error: {e}")
            return "Error while searching."

    def listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio, language="ar-EG").strip()
                    if text:
                        with self.lock:
                            self.history.append(text)
                        # Detect language automatically on first spoken text
                        if self.lang is None:
                            self.lang = detect_language(text)
                        reply = self.generate_reply(text)
                        self.reply_queue.put(reply)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Speech API error: {e}")
                    continue

    def start(self):
        if not self.listening:
            self.listening = True
            self.thread = threading.Thread(target=self.listen_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.listening = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_history(self):
        with self.lock:
            return list(self.history)

    def get_reply_queue(self):
        items = []
        while not self.reply_queue.empty():
            items.append(self.reply_queue.get())
        return items

class SpeechManager:
    def __init__(self):
        self.persons = {}  # {face_id: PersonSpeech}
        self.lock = threading.Lock()

    def add_person(self, face_id):
        with self.lock:
            if face_id in self.persons:
                return self.persons[face_id]
            person = PersonSpeech(face_id)
            self.persons[face_id] = person
            person.start()
            return person

    def stop_all(self):
        for person in self.persons.values():
            person.stop()
        self.persons.clear()