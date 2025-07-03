import streamlit as st
from transformers import pipeline
import torch # Required by transformers for PyTorch backend

# --- Configuration ---
# Set page config for a nicer look and feel
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered", # Centers the content on the page
    initial_sidebar_state="collapsed" # Hides the sidebar by default
)

# --- Load Model (Cached for Performance) ---
# @st.cache_resource is used to cache the model,
# preventing it from being reloaded every time the app re-runs.
# This significantly improves performance.
@st.cache_resource
def load_sentiment_model():
    """
    Loads a pre-trained sentiment analysis model from Hugging Face Transformers.
    The 'distilbert-base-uncased-finetuned-sst-2-english' model is excellent
    for binary (positive/negative) sentiment classification.
    """
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize the sentiment analyzer
sentiment_analyzer = load_sentiment_model()

# --- Custom CSS for Dynamic Background and Element Styling ---
def set_background_color(sentiment_label):
    """
    Applies custom CSS to change the background color of the Streamlit app
    based on the detected sentiment. It also styles other elements for consistency.
    """
    if sentiment_label == "POSITIVE":
        # Light green for positive sentiment
        color_code = "#d4edda"
        text_color = "#155724" # Dark green text for contrast
    elif sentiment_label == "NEGATIVE":
        # Light red for negative sentiment
        color_code = "#f8d7da"
        text_color = "#721c24" # Dark red text for contrast
    else:
        # Default neutral background (light gray)
        color_code = "#f8f9fa"
        text_color = "#212529" # Dark gray text

    # Inject CSS into the Streamlit app.
    # The .stApp class targets the main content area of the Streamlit app.
    # Other classes are for specific Streamlit components (text input, button, markdown).
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color_code};
            color: {text_color};
            transition: background-color 0.5s ease; /* Smooth transition for background change */
        }}
        .stTextInput > div > div > input {{
            border: 2px solid {text_color};
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-size: 1.1rem;
            background-color: white; /* Ensure input background is white */
            color: black; /* Ensure input text is black */
        }}
        .stButton > button {{
            background-color: #007bff; /* A nice blue for the button */
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-size: 1.1rem;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth hover effects */
        }}
        .stButton > button:hover {{
            background-color: #0056b3; /* Darker blue on hover */
            transform: translateY(-2px); /* Slight lift on hover */
        }}
        /* Ensure all text within markdown elements adopts the main text color */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {text_color};
        }}
        .stMarkdown p, .stMarkdown li {{
            color: {text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True # Required to inject raw HTML/CSS
    )

# --- Initial Background (Neutral) ---
# Set a default neutral background color when the app first loads
set_background_color("NEUTRAL")

# --- App Title and Description ---
st.title("💬 Sentiment Analysis App")
st.markdown("""
    Enter any text below and I'll tell you if it's **positive** or **negative**!
    The background color of the app will dynamically change based on the sentiment.
""")

# --- Text Input Area ---
user_input = st.text_area(
    "Enter your text here:",
    height=150, # Set a fixed height for the text area
    placeholder="e.g., I love this new feature! It's absolutely amazing. Or: This is terrible, I hate it."
)

# --- Analyze Button ---
if st.button("Analyze Sentiment"):
    if user_input:
        # Show a spinner while analysis is in progress
        with st.spinner("Analyzing sentiment..."):
            # Perform sentiment analysis using the loaded model
            # The result is a list of dictionaries, so we take the first element [0]
            result = sentiment_analyzer(user_input)[0]
            label = result['label'] # 'POSITIVE' or 'NEGATIVE'
            score = result['score'] # Confidence score

            # Set the background color based on the determined sentiment
            set_background_color(label)

            st.markdown("---") # A horizontal rule for visual separation

            # Display the sentiment result with appropriate styling and emojis
            if label == "POSITIVE":
                st.success(f"**Sentiment: Positive 😊**")
                st.write(f"Confidence: **{score:.2%}**") # Format score as percentage
                st.balloons() # Add a fun balloon animation for positive sentiment
            elif label == "NEGATIVE":
                st.error(f"**Sentiment: Negative 😠**")
                st.write(f"Confidence: **{score:.2%}**")
            else:
                # This case might not be hit often with binary models, but good for robustness
                st.info(f"**Sentiment: Neutral 😐**")
                st.write(f"Confidence: **{score:.2%}**")

            st.markdown("---")
            st.info("Try another text by modifying the input above!")
    else:
        # Warn the user if no text was entered
        st.warning("Please enter some text to analyze.")

# --- Footer ---
st.markdown("""
    <br><br><hr>
    <p style='text-align: center; font-size: 0.8em; color: #6c757d;'>
        Powered by Hugging Face Transformers and Streamlit.
    </p>
    """, unsafe_allow_html=True)
