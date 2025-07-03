import time

# --- Step 1: Create requirements.txt and app.py, then install dependencies ---
# Run this entire block first.

# 1.1 Create requirements.txt
requirements_content = """
streamlit==1.36.0
transformers==4.42.1
torch==2.3.1
"""
with open("requirements.txt", "w") as f:
    f.write(requirements_content)
print("Created requirements.txt")

# 1.2 Create app.py
# Writing the content programmatically helps avoid the 'invalid character' SyntaxError.
app_py_content = """
import streamlit as st
from transformers import pipeline
import torch

# --- Configuration ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Model (Cached for Performance) ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# --- Custom CSS for Dynamic Background and Element Styling ---
def set_background_color(sentiment_label):
    if sentiment_label == "POSITIVE":
        color_code = "#d4edda"
        text_color = "#155724"
    elif sentiment_label == "NEGATIVE":
        color_code = "#f8d7da"
        text_color = "#721c24"
    else:
        color_code = "#f8f9fa"
        text_color = "#212529"

    st.markdown(
        f\"\"\"
        <style>
        .stApp {{
            background-color: {color_code};
            color: {text_color};
            transition: background-color 0.5s ease;
        }}
        .stTextInput > div > div > input {{
            border: 2px solid {text_color};
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-size: 1.1rem;
            background-color: white;
            color: black;
        }}
        .stButton > button {{
            background-color: #007bff;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-size: 1.1rem;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease, transform 0.2s ease;
        }}
        .stButton > button:hover {{
            background-color: #0056b3;
            transform: translateY(-2px);
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {text_color};
        }}
        .stMarkdown p, .stMarkdown li {{
            color: {text_color};
        }}
        </style>
        \"\"\",
        unsafe_allow_html=True
    )

set_background_color("NEUTRAL")

st.title("💬 Sentiment Analysis App")
st.markdown(\"\"\"
    Enter any text below and I'll tell you if it's **positive** or **negative**!
    The background color of the app will dynamically change based on the sentiment.
\"\"\")

user_input = st.text_area(
    "Enter your text here:",
    height=150,
    placeholder="e.g., I love this new feature! It's absolutely amazing. Or: This is terrible, I hate it."
)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            result = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")(user_input)[0]
            label = result['label']
            score = result['score']

            set_background_color(label)

            st.markdown("---")

            if label == "POSITIVE":
                st.success(f"**Sentiment: Positive 😊**")
                st.write(f"Confidence: **{score:.2%}**")
                st.balloons()
            elif label == "NEGATIVE":
                st.error(f"**Sentiment: Negative 😠**")
                st.write(f"Confidence: **{score:.2%}**")
            else:
                st.info(f"**Sentiment: Neutral 😐**")
                st.write(f"Confidence: **{score:.2%}**")

            st.markdown("---")
            st.info("Try another text by modifying the input above!")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown(\"\"\"
    <br><br><hr>
    <p style='text-align: center; font-size: 0.8em; color: #6c757d;'>
        Powered by Hugging Face Transformers and Streamlit.
    </p>
    \"\"\", unsafe_allow_html=True)
"""
with open("app.py", "w") as f:
    f.write(app_py_content)
print("Created app.py")

# 1.3 Install Dependencies
print("\nInstalling dependencies (this may take a few minutes)...")
!pip install -r requirements.txt -q
print("Dependencies installed!")

# 1.4 Install ngrok for public access (optional, but useful in Colab)
print("\nInstalling pyngrok...")
!pip install pyngrok -q
print("pyngrok installed!")

# Add a small delay to ensure all installations are registered
print("\nWaiting a few seconds for installations to settle...")
time.sleep(5) # Wait for 5 seconds
print("Ready to launch!")

print("\n--- Setup Complete! ---")
print("Now, run the next cell to launch your Streamlit app.")
```


```python
# --- Step 2: Run the Streamlit App ---
# IMPORTANT: Run this cell *after* the previous setup cell has completed.
# This command will start the Streamlit server and provide a public URL via ngrok.
# Look for a URL like 'https://something-random.loca.lt' in the output below.
# It might take a moment for the URL to appear.

!streamlit run app.py & npx localtunnel --port 8501
