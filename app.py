import streamlit as st
import cv2
import time
import wikipedia
import atexit
from datetime import datetime
from utils.camera import CameraThread
from utils.speech import SpeechManager
from utils.visualization import plot_speech_counts

st.set_page_config(page_title="VSI", layout="wide")
st.title("VisuoSpeech Intelligence (VSI)")

wikipedia.set_lang("ar")

if "camera" not in st.session_state:
    st.session_state.camera = CameraThread()
    st.session_state.camera.start()
    atexit.register(lambda: st.session_state.camera.stop())

if "speech_manager" not in st.session_state:
    st.session_state.speech_manager = SpeechManager()

if "spoken_history" not in st.session_state:
    st.session_state.spoken_history = {}
if "spoken_replies" not in st.session_state:
    st.session_state.spoken_replies = {}
if "paused" not in st.session_state:
    st.session_state.paused = False

frame_placeholder = st.empty()
ids_placeholder = st.empty()
speech_placeholder = st.container()

st.sidebar.header("🎛 Controls")
if st.sidebar.button("⏸ Pause / ▶ Resume Camera"):
    st.session_state.paused = not st.session_state.paused
    st.sidebar.write("Camera Paused" if st.session_state.paused else "Camera Running")

if st.sidebar.button("📊 Show Speech Analysis"):
    st.header("Speech Analysis")
    plot_speech_counts(st.session_state.spoken_history)

camera = st.session_state.camera
speech_manager = st.session_state.speech_manager

st.markdown("---")
st.subheader("🎥 Live Feed and 🗣 Speech Recognition")

frame_update_interval = 0.1

while True:
    if not st.session_state.paused:
        frame, faces = camera.get_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            current_ids = list(faces.keys())
            ids_placeholder.markdown(f"**Detected IDs:** {current_ids}")

            disappeared_ids = [fid for fid in list(speech_manager.persons.keys()) if fid not in current_ids]
            for fid in disappeared_ids:
                speech_manager.persons[fid].stop()
                speech_manager.persons.pop(fid, None)

            for fid in current_ids:
                if fid not in speech_manager.persons:
                    speech_manager.add_person(fid)
                    st.session_state.spoken_history.setdefault(fid, [])
                    st.session_state.spoken_replies.setdefault(fid, [])

                person = speech_manager.persons[fid]
                new_history = person.get_history()
                old_history = st.session_state.spoken_history.get(fid, [])

                if len(new_history) > len(old_history):
                    new_text = new_history[-1]
                    st.session_state.spoken_history[fid] = new_history

                    reply = person.generate_reply(new_text)
                    st.session_state.spoken_replies[fid].append((datetime.now(), reply))

                    time_str = datetime.now().strftime('%I:%M:%S %p').lstrip("0")

                    with speech_placeholder:
                        st.subheader(f"ID {fid}")
                        st.write(f"[{time_str}] 🎙 Said: {new_text}")
                        st.success(f"[{time_str}] 📚 Wikipedia: {reply}")

    time.sleep(frame_update_interval)
