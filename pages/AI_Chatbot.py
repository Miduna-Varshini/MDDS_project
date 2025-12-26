import streamlit as st
import openai

# ================== Page config ==================
st.set_page_config(page_title="AI Health Assistant", page_icon="🤖")
st.title("🤖 AI Health Assistant")
st.write("Ask questions about symptoms, diseases, reports, or prevention.")

# ================== Load API key ==================
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in secrets.toml")
    st.stop()

openai.api_key = api_key

# ================== Initialize session ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================== Display chat history ==================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================== User input ==================
user_input = st.chat_input("Ask your health question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ================== Call OpenAI GPT ==================
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change to gpt-4 if you have access
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content

    except Exception as e:
        reply = f"⚠️ Error: {e}"

    # ================== Save and display reply ==================
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
