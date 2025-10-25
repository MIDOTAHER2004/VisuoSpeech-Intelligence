import pandas as pd
import streamlit as st

def plot_speech_counts(spoken_history):
    counts = {fid: len(history) for fid, history in spoken_history.items()}
    
    if not counts:
        st.warning("⚠️ No speech data yet.")
        return

    df = pd.DataFrame(list(counts.items()), columns=["ID", "Sentences Count"])

    df = df.sort_values(by="Sentences Count", ascending=False).reset_index(drop=True)

    st.subheader("📋 Speech Count Table")
    st.table(df)

    st.subheader("📊 Speech Activity by ID")
    st.bar_chart(df.set_index("ID"))
