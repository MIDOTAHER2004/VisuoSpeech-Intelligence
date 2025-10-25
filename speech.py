import speech_recognition as sr
import threading
from queue import Queue
import wikipedia

wikipedia.set_lang("ar")

class PersonSpeech:
    def __init__(self, face_id):
        self.face_id = face_id
        self.history = []
        self.reply_queue = Queue()
        self.lock = threading.Lock()
        self.listening = False
        self.recognizer = sr.Recognizer()
        self.thread = None

    def generate_reply(self, text):
        try:
            summary = wikipedia.summary(text, sentences=2)
            return summary
        except wikipedia.DisambiguationError as e:
            return f"Ambiguous question, try specifying more: {e.options[:5]}"
        except wikipedia.PageError:
            return "No information found on this topic."
        except Exception:
            return "Error occurred while searching."

    def listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=None)
                    try:
                        text = self.recognizer.recognize_google(audio, language="ar-EG").strip()
                    except sr.UnknownValueError:
                        continue  

                    if text:
                        with self.lock:
                            self.history.append(text)
                        reply = self.generate_reply(text)
                        self.reply_queue.put(reply)

                except sr.RequestError:
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
        self.persons = {}
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
