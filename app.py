import streamlit as st
import cv2
import time
import wikipedia
from datetime import datetime
from utils.camera import CameraThread
from utils.speech import SpeechManager
from utils.visualization import plot_speech_counts

# Page setup
st.set_page_config(page_title="VSI", layout="wide")
st.title("VisuoSpeech Intelligence")

# Set Wikipedia language to Arabic
wikipedia.set_lang("ar")

# Start camera once
if "camera" not in st.session_state:
    st.session_state["camera"] = CameraThread()
    st.session_state["camera"].start()

# Initialize session state variables for history, replies, and pause
if "spoken_history" not in st.session_state:
    st.session_state["spoken_history"] = {}
if "spoken_replies" not in st.session_state:
    st.session_state["spoken_replies"] = {}
if "paused" not in st.session_state:
    st.session_state["paused"] = False

# Placeholders for Streamlit display
camera = st.session_state["camera"]
frame_placeholder = st.empty()
ids_placeholder = st.empty()
speech_placeholder = st.container()

# Create or get speech manager
if "speech_manager" not in st.session_state:
    st.session_state["speech_manager"] = SpeechManager()
speech_manager = st.session_state["speech_manager"]

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Pause/Resume Camera"):
    st.session_state["paused"] = not st.session_state["paused"]
    st.sidebar.write("Camera Paused" if st.session_state["paused"] else "Camera Running")

# Analysis button
show_analysis = st.sidebar.button("Show Analysis")

# Display analysis table once if button pressed
if show_analysis:
    st.header("Analysis")
    plot_speech_counts(st.session_state["spoken_history"], st)

# Main loop to update frames and speech
while True:
    if not st.session_state["paused"]:
        frame, faces = camera.get_frame()
        if frame is not None:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Display current detected IDs
            current_ids = list(faces.keys())
            ids_placeholder.write(f"Current IDs: {current_ids}")

            # Add new IDs to speech manager
            for fid in current_ids:
                if fid not in speech_manager.persons:
                    person = speech_manager.add_person(face_id=fid)
                    st.session_state["spoken_history"][fid] = []
                    st.session_state["spoken_replies"][fid] = []

            # Update speech history and replies
            with speech_placeholder:
                for fid in current_ids:
                    person = speech_manager.persons[fid]
                    new_history = person.get_history()
                    old_history = st.session_state["spoken_history"].get(fid, [])
                    last_text = None

                    if len(new_history) > len(old_history):
                        last_text = new_history[-1]
                        st.session_state["spoken_history"][fid] = new_history

                    last_reply = None
                    if last_text:
                        # Generate reply using PersonSpeech method
                        reply = person.generate_reply(last_text)
                        st.session_state["spoken_replies"][fid].append((datetime.now(), reply))
                        last_reply = (datetime.now(), reply)

                    # Display last speech and Wikipedia reply
                    if last_text or last_reply:
                        st.subheader(f"ID {fid}")
                        if last_text:
                            st.write(f"[{datetime.now().strftime('%I:%M:%S %p')}] Said: {last_text}")
                        if last_reply:
                            st.success(f"[{last_reply[0].strftime('%I:%M:%S %p')}] Wikipedia: {last_reply[1]}")

    time.sleep(0.1)