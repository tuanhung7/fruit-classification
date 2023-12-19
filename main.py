import io
import os

import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from keras.models import Model, load_model
from keras.utils import img_to_array, load_img
from PIL import Image

# Load the model and labels
model = load_model('87.h5')
labels = {
    0: "apple",
    1: "avocado",
    2: "banana",
    3: "cucumber",
    4: "dragonfruit",
    5: "durian",
    6: "grape",
    7: "guava",
    8: "kiwi",
    9: "lemon",
    10: "lychee",
    11: "mango",
    12: "orange",
    13: "papaya",
    14: "pear",
    15: "pineapple",
    16: "pomegranate",
    17: "strawberry",
    18: "tomato",
    19: "watermelon",
}


def resize_image(img_path, size=(224, 224)):
    """This function resize the image to square shape and save it to the same path."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = (
        cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
    )

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[:h, :w, :]
    mask = cv2.resize(mask, size, interpolation)
    cv2.imwrite(img_path, mask)


def fetch_calories(prediction):
    try:
        calories_data = {
            "apple": "52.1",
            "avocado": "160",
            "banana": "89",
            "cucumber": "15",
            "dragonfruit": "60",
            "durian": "149",
            "grape": "69",
            "guava": "68",
            "kiwi": "61",
            "lemon": "29",
            "lychee": "66",
            "mango": "60",
            "orange": "43",
            "papaya": "43",
            "pear": "57",
            "pineapple": "50",
            "pomegranate": "83",
            "strawberry": "32",
            "tomato": "18",
            "watermelon": "30",
        }
        calories = calories_data.get(prediction)

        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img(img_path):
    # Load the image
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict the image
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    predicted_label = labels[predicted_class]
    return predicted_label, prediction[0][predicted_class] * 100

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x

def GradCAM_explain(image_path, model, predicted_class):
    image = resize_image(image_path)
    preprocessed_image = preprocess_image(image_path)

    last_conv_layer = "conv2d_13"
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(preprocessed_image)
        loss = preds[:, predicted_class]
    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    grads *= weights
    grads = tf.reduce_sum(grads, axis=(0, 3))
    grads = tf.nn.relu(grads)
    grads /= tf.reduce_max(grads)
    grads = tf.cast(grads * 255.0, 'uint8')
    cam = np.array(Image.fromarray(grads.numpy()).resize((224, 224)))
    orig_image = image
    cam_rgb = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_rgb_resized = cv2.resize(cam_rgb, (orig_image.shape[1], orig_image.shape[0]))
    alpha = 0.5
    overlay = orig_image.copy()
    cv2.addWeighted(cam_rgb_resized, alpha, overlay, 1 - alpha, 0, overlay)
    file_path = hash_image_after_write(overlay)
    return file_path

def generate_hash(file_content):
    hash_object = hashlib.sha256()
    hash_object.update(file_content)
    return hash_object.hexdigest()

def hash_image_after_write(image):
    # Tạo giá trị hash cho nội dung của file và lưu ảnh
    image_content = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
    hash_filename = generate_hash(image_content)
    file_path = f"explain/{hash_filename}.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return file_path

should_explain = False

def process_image(img_content):
    try:
        img_hash = hashlib.md5(img_content).hexdigest()
        img = Image.open(io.BytesIO(img_content))
        st.image(img, use_column_width=False, width=500)

        save_image_path = os.path.join('images', f'{img_hash}.png')
        with open(save_image_path, "wb") as f:
            f.write(img_content)

        resize_image(save_image_path)

        result, percentage = processed_img(save_image_path)

        st.success("**Predicted : " + result + '**')
        st.info('**Accuracy : ' + str(round(percentage, 2)) + '%**')

        cal = fetch_calories(result)
        if cal:
            st.warning('**Calories : ' + cal + ' calories in 100 grams**')
        predicted_class = list(labels.values()).index(result)
        print("save_image_path:", save_image_path)
        explanation_img_path = GradCAM_explain(save_image_path, model, predicted_class)
        st.image(Image.open(explanation_img_path), use_column_width=False, width=500,
                 caption="GradCAM Visualization")
    except Exception as e:
        st.error("Can't process the image. Please check the image and try again.")
        print(e)

def process_uploaded_image(img_file):
    img_content = img_file.read()
    process_image(img_content)

def process_url_image(img_url):
    try:
        img_content = requests.get(img_url).content
        process_image(img_content)

    except Exception as e:
        st.error("Can't process the image from the provided URL. Please check the URL and try again.")
        print(e)

def run():
    st.title("Fruit Recognition")
    st.sidebar.title("Choose Image Source")

    img_option = st.sidebar.radio(" ", ["Upload Image", "Paste Image URL"])

    if img_option == "Upload Image":
        img_file = st.sidebar.file_uploader("Choose an Image", type=["jpg", "png", "webp"])
        if img_file is not None:
            process_uploaded_image(img_file)

    elif img_option == "Paste Image URL":
        img_url = st.sidebar.text_input("Paste Image URL")
        if st.sidebar.button("Process Image", key="process_image_button", help="Click to process the image"):
            process_url_image(img_url)


run()
