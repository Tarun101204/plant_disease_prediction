import streamlit as st
from PIL import Image
from utils import predict_image, chat_with_ai , generate_info

import time 

def type_text(text):
    placeholder = st.empty()
    typed = ""
    
    speed = 0.001 if len(text) > 300 else 0.003
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(speed)
    
    return typed

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = None

if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}

if "disease" not in st.session_state:
    st.session_state.disease = None

if "image" not in st.session_state:
    st.session_state.image = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# ---------------- UI ----------------
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱")

st.markdown("""
# 🌱 Plant Disease Detection
Upload a leaf image and get instant disease prediction with treatment suggestions.
""")

# Reset
if st.button("🔄 Reset"):
    st.session_state.results = None
    st.session_state.disease = None
    st.session_state.image = None
    st.session_state.chat_history = {}
    st.rerun()


# ---------------- IMAGE INPUT ----------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file)


# ---------------- PREDICT ----------------
if st.session_state.image is not None:

    if st.button(" Predict Disease"):

        with st.spinner("Analyzing leaf..."):

            results = predict_image(st.session_state.image)

        st.session_state.results = results
        disease = results[0][1]
        st.session_state.disease = disease
        if disease not in st.session_state.chat_history:
            st.session_state.chat_history[disease] = []


# ---------------- DISPLAY RESULTS ----------------
if st.session_state.results:

    results = st.session_state.results

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(st.session_state.image, use_container_width=True)

    with col2:
        best_disease = results[0][1]
        best_conf = results[0][2]

        st.success(f"🌿 Most Likely: {best_disease} ({best_conf:.2f}%)")

    st.markdown("---")

    st.markdown("## AI Explanation")
    disease = st.session_state.disease
    if disease: 
        if disease in st.session_state.ai_cache:
            ai_response = st.session_state.ai_cache[disease]
            type_text(ai_response)
        else:
            generate = st.button("Generate AI Explanation")
            if generate:
                with st.spinner("Generating explanation..."):
                    ai_response = generate_info(disease)
                st.session_state.ai_cache[disease] = ai_response
            if disease in st.session_state.ai_cache:
                type_text(st.session_state.ai_cache[disease])


# ---------------- CHAT SECTION ----------------
st.markdown("---")
st.markdown("## 💬 Ask About the Disease")

if st.session_state.disease:

    disease = st.session_state.disease

    if disease not in st.session_state.chat_history:
        st.session_state.chat_history[disease] = []

    current_chat = st.session_state.chat_history[disease]

    for role,message in current_chat:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

    user_input = st.text_input("Ask a question:", key="chat_input")
    send = st.button("Ask AI")

    if user_input and send:

        current_chat.append(("user", user_input))

        with st.spinner("Thinking..."):
            reply = chat_with_ai(disease, user_input)  

        with st.chat_message("assistant"):
            response = type_text(reply)

        current_chat.append(("ai",response))
        
else:
    st.info("Upload and predict an image first.")