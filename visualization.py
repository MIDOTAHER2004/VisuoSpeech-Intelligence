import pandas as pd

def plot_speech_counts(spoken_history, st):
    counts = {fid: len(history) for fid, history in spoken_history.items()}
    if not counts:
        st.warning("No speech data yet.")
        return
    
    # Convert dict to DataFrame
    df = pd.DataFrame(list(counts.items()), columns=["ID", "Sentences Count"])
    
    # Sort by sentence count in descending order
    df = df.sort_values(by="Sentences Count", ascending=False).reset_index(drop=True)
    
    st.table(df)