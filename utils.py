import numpy as np
import tensorflow as tf
import json
import google.generativeai as genai
from PIL import Image

def generate_info(disease_name):
    prompt = f"""
    You are a plant disease expert.

    Disease: {disease_name}

    Give: 
    1.Description
    2.Treatment (practical steps)

    """
    response = model_ai.generate_content(prompt)
    return response.text

def chat_with_ai(disease_name, user_query):
    prompt = f"""
    You are a plant disease expert.
    
    Disease: {disease_name}

    User question: {user_query}

    Answer clearly and simple.
    """

    response = model_ai.generate_content(prompt)
    return response.text


#GEN AI
genai.configure(api_key="AIzaSyBCL1jtVGGv7Q3O5Yf8oqahM_s1NZYykGw")
model_ai = genai.GenerativeModel("models/gemini-2.5-flash")

# Load model
model = tf.saved_model.load("model/plant_model")
infer = model.signatures["serving_default"]

# Load class mapping
with open("model/class_indices.json") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# Load disease info
with open("disease_info.json") as f:
    disease_info = json.load(f)


def clean_name(name):
    return name.replace("___", " ").replace("_", " ")


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((256,256))

    # center crop
    width, height = image.size
    left = (width - 224)//2
    top = (height - 224)//2
    right = left + 224
    bottom = top + 224

    image = image.crop((left, top, right, bottom))

    img = np.array(image) / 255.0
    img_array = np.expand_dims(img, axis=0).astype(np.float32)

    prediction = infer(tf.constant(img_array))
    prediction = list(prediction.values())[0].numpy()

    # 🔥 Top 3 predictions
    top_indices = prediction[0].argsort()[-3:][::-1]

    results = []

    for i in top_indices:
        disease_class = index_to_class[i]
        disease_name = clean_name(disease_class)
        confidence = float(prediction[0][i]) * 100

        results.append((disease_class, disease_name, confidence))

    # ✅ Get info for top prediction only
    return results
