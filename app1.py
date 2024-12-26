import streamlit as st
import base64
import openai
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input

openai.api_key =  st.secrets["OPENAI_API_KEY"]

class_labels = [
    "Algae",
    "Bubbles and blisters",
    "Cracks",
    "Efflorescence",
    "Fungus",
    "Patchiness",
    "Peeling",
    "Poor Hiding",
    "Shade Variation"
]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@st.cache_resource
def load_trained_model():
    return load_model('my_model12.h5')

loaded_model = load_trained_model()

st.title("Wall Defect Classification and AI Analysis")
st.write("Upload an image to classify wall defects and generate AI-based descriptions.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Read and preprocess the input image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    input_img_resized = cv2.resize(input_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    x = img_to_array(input_img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = loaded_model.predict(x)

    threshold = 0.3

    class_indices = np.where(preds[0] > threshold)[0]
    class_probabilities = preds[0][class_indices]

    results_text = ""
    predicted_defects = []
    if len(class_indices) > 0:
        for i, class_idx in enumerate(class_indices):
            class_name = class_labels[class_idx]
            results_text += f"{class_name} (Class {class_idx}): Probability {class_probabilities[i]:.2f}\n"
            predicted_defects.append(class_name)
    else:
        results_text = "No classes detected with a probability greater than the threshold."

    # Display classification results in a text box
    st.text_area("Classification Results:", value=results_text, height=200)

    # Encode the uploaded image as Base64
    base64_image = base64.b64encode(file_bytes).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{base64_image}"

    # Generate AI-based descriptions using OpenAI API
    if predicted_defects:
        defects_string = ", ".join(predicted_defects)
        ai_prompt = (
            f"Our trained model predicts the following defects: {defects_string}. "
            f"Can you analyze the following image and generate AI-based descriptions for these defects?"
        )

        st.write("Analyzing image")
        try:
            response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
               {
                "role": "user",
                "content": [
                    {"type": "text", 
                     "text":ai_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                 ],
             }
           ],
             max_tokens=300,

    )
            # Extract AI-generated descriptions
            ai_description = response.choices[0].message.content
            st.text_area("AI-Generated Description:", value=ai_description, height=200)

        except Exception as e:
            st.error(f"An error occurred while generating AI-based descriptions: {str(e)}")
    else:
        st.warning("No defects detected. AI analysis skipped.")
