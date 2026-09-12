import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Dense, Reshape, Multiply, Input, Add, Lambda, Concatenate, Activation, Rescaling)
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify, render_template
import base64

# --- Constants ---
IMG_SIZE = 256
NUM_CLASSES = 3
DROPOUT_RATE = 0.40
CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]
USE_CHANNEL_ATTENTION = True
USE_SPATIAL_ATTENTION = True

# Short, human-readable notes shown next to the Grad-CAM heatmap.
# Keep these clinically neutral -- they describe what the model attended to,
# not a diagnostic claim.
CLASS_NOTES = {
    "Bleeding": "The model focused on hyperdense (brighter) regions typical of "
                "acute hemorrhage. The highlighted area shows where this evidence was found.",
    "Ischemia": "The model focused on hypodense (lower-attenuation) regions "
                "typical of ischemic changes. The highlighted area shows where this evidence was found.",
    "Normal": "No focal hemorrhagic or ischemic pattern stood out. Attention is "
              "spread diffusely across the scan rather than concentrated in one region.",
}

MIN_CONFIDENT_SCORE = 60.0  # below this, we flag the prediction as low-confidence

# --- MT-Net Architecture ---
def se_block(input_tensor, reduction=16, block_id=1):
    channels = int(input_tensor.shape[-1])
    reduced_channels = max(channels // reduction, 4)
    se = GlobalAveragePooling2D(name=f"se_gap_{block_id}")(input_tensor)
    se = Reshape((1, 1, channels), name=f"se_reshape_{block_id}")(se)
    se = Dense(reduced_channels, activation="relu", name=f"se_reduce_{block_id}")(se)
    se = Dense(channels, activation="sigmoid", name=f"se_expand_{block_id}")(se)
    return Multiply(name=f"se_multiply_{block_id}")([input_tensor, se])

def spatial_attention_block(input_tensor, block_id=1):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), name=f"spatial_avg_pool_{block_id}")(input_tensor)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True), name=f"spatial_max_pool_{block_id}")(input_tensor)
    concat = Concatenate(axis=-1, name=f"spatial_concat_{block_id}")([avg_pool, max_pool])
    spatial = Conv2D(1, (7, 7), padding="same", activation="sigmoid", name=f"spatial_attention_conv_{block_id}")(concat)
    return Multiply(name=f"spatial_attention_multiply_{block_id}")([input_tensor, spatial])

def residual_sc_block(x, filters, block_id=1):
    shortcut = x
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"shortcut_conv_{block_id}")(shortcut)
        shortcut = BatchNormalization(name=f"shortcut_bn_{block_id}")(shortcut)

    x = Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"res_conv1_{block_id}")(x)
    x = BatchNormalization(name=f"res_bn1_{block_id}")(x)
    x = Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"res_conv2_{block_id}")(x)
    x = BatchNormalization(name=f"res_bn2_{block_id}")(x)

    if USE_CHANNEL_ATTENTION: x = se_block(x, reduction=16, block_id=block_id)
    if USE_SPATIAL_ATTENTION: x = spatial_attention_block(x, block_id=block_id)

    x = Add(name=f"res_add_{block_id}")([shortcut, x])
    x = Activation("relu", name=f"res_output_relu_{block_id}")(x)
    return MaxPooling2D((2, 2), name=f"res_pool_{block_id}")(x)

def build_mt_net(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
    inputs = Input(shape=input_shape, name="input_image")
    x = Rescaling(1.0 / 255.0, name="input_rescaling")(inputs)
    x = Conv2D(32, (3, 3), padding="same", activation="relu", name="initial_conv")(x)
    x = BatchNormalization(name="initial_bn")(x)
    x = MaxPooling2D((2, 2), name="initial_pool")(x)
    x = residual_sc_block(x, 64, block_id=1)
    x = residual_sc_block(x, 128, block_id=2)
    x = residual_sc_block(x, 256, block_id=3)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dense(256, activation="relu", name="classifier_dense")(x)
    x = Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="classification_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="MT_Net")

# --- Load Model ---
print("Loading MT-Net Architecture and Weights...")
model = build_mt_net()
# ওয়েটস ফাইলের নাম আপনার সেভ করা নামের সাথে মিলিয়ে নিন
model.load_weights("best_mt_net_stage2.weights.h5")
# model.load_weights("best_mt_net_stage2_weights.h5")
print("Model loaded successfully!")

# --- Web App Helpers ---
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_web(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian_3 = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(bilateral, 1.5, gaussian_3, -0.5, 0)
    final_img = cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2BGR)
    final_img = cv2.resize(final_img, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(final_img, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="res_output_relu_3", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# --- Flask Server ---
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload a PNG or JPG image.'}), 400
    try:
        image_bytes = file.read()
        processed_img_array = preprocess_image_for_web(image_bytes)
        preds = model.predict(processed_img_array)[0]

        pred_results = [{'class': name, 'confidence': round(float(p) * 100, 2)} for name, p in zip(CLASS_NAMES, preds)]
        pred_results = sorted(pred_results, key=lambda x: x['confidence'], reverse=True)
        top_class_name = pred_results[0]['class']
        top_class_index = CLASS_NAMES.index(top_class_name)
        top_confidence = pred_results[0]['confidence']

        heatmap = make_gradcam_heatmap(processed_img_array, model, pred_index=top_class_index)

        nparr = np.frombuffer(image_bytes, np.uint8)
        original_cv_img = cv2.resize(cv2.imdecode(nparr, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        heatmap_resized = np.uint8(255 * cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)))
        heatmap_colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_cv_img, 0.6, heatmap_colormap, 0.4, 0)

        orig_base64 = base64.b64encode(cv2.imencode('.jpg', original_cv_img)[1]).decode('utf-8')
        heatmap_base64 = base64.b64encode(cv2.imencode('.jpg', superimposed_img)[1]).decode('utf-8')

        return jsonify({
            'predictions': pred_results,
            'original_img': orig_base64,
            'gradcam_img': heatmap_base64,
            'top_class': top_class_name,
            'note': CLASS_NOTES.get(top_class_name, ''),
            'low_confidence': top_confidence < MIN_CONFIDENT_SCORE,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     print("Starting Localhost Server...")
#     app.run(port=5000, debug=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)