from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import matplotlib.pyplot as plt

model_path = 'saved_model'  # Adjust this path as necessary
weights_path = 'saved_model/saved_weights.h5'

model = load_model(model_path)
model.load_weights(weights_path)

app = Flask(__name__)

# Function to get the Grad-CAM heatmap
def get_grad_cam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_grad_cam(img_array, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = img_array[0]
    if np.max(img) > 1:
        img = img / 255.0

    # Superimpose the heatmap on the image
    superimposed_img = heatmap * 0.004 + img * (1 - alpha)
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))

    # Convert BGR (OpenCV format) to RGB (matplotlib format)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img

@app.route('/', methods=['GET'])
def hello():
    return render_template("test.html")

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    if imagefile:
        image_path = "./images/" + imagefile.filename
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        imagefile.save(image_path)
        image = Image.open(image_path)
        image = np.array(image)

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = image

        img_resized = resize(gray_img, (128, 128), mode='constant')
        processed_image = np.expand_dims(img_resized, axis=-1)
        processed_image = np.expand_dims(processed_image, axis=0)  # Adding batch dimension
        class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        classification = f'The predicted kidney condition is: {predicted_class}'

        # Generate Grad-CAM heatmap
        heatmap = get_grad_cam_heatmap(model, processed_image, last_conv_layer_name='conv2d_27')
        superimposed_img = display_grad_cam(processed_image, heatmap)
        heatmap_path = "./static/heatmap.jpg"
        cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    else:
        classification = 'No image uploaded.'
        heatmap_path = None

    return render_template('test.html', prediction=classification, heatmap=heatmap_path)

if __name__ == "__main__":
    app.run(debug=True, port=3000)
