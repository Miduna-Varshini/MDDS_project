import streamlit as st
import google.generativeai as genai

# Page config
st.set_page_config(page_title="AI Health Assistant", page_icon="🤖")

st.title("🤖 AI Health Assistant")
st.write("Ask questions about symptoms, diseases, reports, or prevention.")

# Load API key from secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in secrets.toml")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)

# Use a valid Gemini model
model = genai.GenerativeModel("models/gemini-2.0-flash")  # Valid model

# Initialize chat session in Streamlit state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask your health question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = st.session_state.chat.send_message(user_input)
        reply = response.text
    except genai.ErrorQuotaExceeded:
        reply = "⚠️ Quota exceeded. Please try again later or upgrade your plan."
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
