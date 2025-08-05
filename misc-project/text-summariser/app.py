import streamlit as st
from text_summariser import summarise_text

st.set_page_config(page_title="AI Text Summarsier", layout="wide")

st.title("📝 AI-Powered Text Summariser")
st.write("Paste in some text below, pick how many sentences you want back, and hit Summarise.")

# 1. Text input
user_input = st.text_area(
    label="Please Enter your text here",
    height=300,
    placeholder="Type or paste the article, report, etc..."
)

# 2. Sentence count selector
n_sentences = st.slider(
    label="Number of summary sentences",
    min_value=1,
    max_value=10,
    value=3
)

# 3. Summarise button
if st.button("🔎 Summarise"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("🧠 Summarising…"):
            summary = summarise_text(user_input, sentence_count=n_sentences)
        st.subheader("Summary")
        st.write(summary)