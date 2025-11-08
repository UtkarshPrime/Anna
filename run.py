import streamlit as st
from main import chat_me
from sound import speak_text, listen_and_transcribe, stop_speaking
from PIL import Image

# ---- Page Settings ----
st.set_page_config(page_title="Anna", page_icon="🦇", layout="wide")
st.markdown("<h1 style='text-align: center;'>🦇 ANNA</h1>", unsafe_allow_html=True)

# ---- Session State ----
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_input_type" not in st.session_state:
        st.session_state.last_input_type = None
init_session_state()

# ---- Layout ----
left_col, right_col = st.columns([1, 2])

# ---- Left: Image ----
with left_col:
    st.markdown("<h3 style='text-align: center;'>Anna Image</h3>", unsafe_allow_html=True)
    image = Image.open("Anna.png")
    st.image(image, use_container_width=True)

# ---- Right: Chat and Input ----
with right_col:
    st.markdown("<h3 style='text-align: center;'>Chatbot Responses</h3>", unsafe_allow_html=True)

    # Show chat history
    chat_box = st.container()
    for i, (speaker, msg) in enumerate(st.session_state.chat_history):
        with chat_box:
            with st.chat_message(speaker):
                st.markdown(msg)
                if speaker == "Anna":
                    if st.button(f"🔊 Speak {i}", key=f"speak_{i}"):
                        try:
                            speak_text(msg)
                        except Exception as e:
                            st.error(f"Failed to speak: {e}")

    # ---- Input Form ----
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", placeholder="Whisper your secrets...")
        col1, col2 = st.columns([1, 1])
        with col1:
            speak = st.form_submit_button("🎤 Speak")
        with col2:
            send = st.form_submit_button("📩 Send")

    # ---- Text Input Handling ----
    if send and user_input:
        response = chat_me(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Anna", response))
        st.session_state.last_input_type = "text"
        st.rerun()

    # ---- Voice Input Handling ----
    if speak:
        stop_speaking()  # Stop any ongoing voice output first
        transcribed = listen_and_transcribe()
        if transcribed:
            response = chat_me(transcribed)
            st.session_state.chat_history.append(("You", transcribed))
            st.session_state.chat_history.append(("Anna", response))
            #speak_text(response)
            st.session_state.last_input_type = "voice"
            st.rerun()

    # ---- Speak response automatically if last input was voice ----
    if (
        st.session_state.last_input_type == "voice"
        and st.session_state.chat_history
        and st.session_state.chat_history[-1][0] == "Anna"
    ):
        try:
            speak_text(st.session_state.chat_history[-1][1])
        except Exception as e:
            st.error(f"Failed to auto-speak: {e}")
        finally:
            st.session_state.last_input_type = None  # ✅ Always reset
