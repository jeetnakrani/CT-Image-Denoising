
import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import zipfile
import io
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from pydicom import dcmread
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load pre-trained autoencoder model
MODEL_PATH = "models/autoencoder_noise.h5"
if os.path.exists(MODEL_PATH):
    autoencoder = load_model(MODEL_PATH)
    print("Model loaded. Expected input shape:", autoencoder.input_shape)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please check the path.")

def calculate_original_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(signal / noise)

def calculate_denoised_snr(original, denoised):
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - denoised) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def load_image(file):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext == ".dcm":
        dicom = dcmread(file)
        img = dicom.pixel_array.astype(np.float32)
        if len(img.shape) == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    else:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR).astype(np.float32)
    img = cv2.resize(img, (200, 200))
    img = img / 255.0
    return img

def denoise_image(model, image):
    image_expanded = np.expand_dims(image, axis=0)
    denoised_image = model.predict(image_expanded)
    return np.squeeze(denoised_image)

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', (image * 255).astype(np.uint8))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def overlay_snr(image, snr_text):
    img_uint8 = (image * 255).astype(np.uint8)
    h, w, _ = img_uint8.shape
    new_h = h + 30
    new_img = np.zeros((new_h, w, 3), dtype=np.uint8)
    new_img[:h, :, :] = img_uint8
    cv2.putText(new_img, f"SNR: {snr_text} dB", (w//2 - 50, h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return new_img / 255.0

# Single image endpoints
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/service")
def service():
    return render_template("service.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/appointment")
def appointment():
    return render_template("appointment.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    original_image = load_image(file)
    original_snr = calculate_original_snr(original_image[..., 0])
    original_base64 = image_to_base64(original_image)
    app.config['original_image'] = original_image
    return jsonify({
        "original_image": original_base64,
        "original_snr": round(float(original_snr), 2) if original_snr != float('inf') else "∞"
    })

@app.route("/denoise", methods=["POST"])
def denoise():
    if "image" not in request.json:
        return jsonify({"error": "No image data provided"}), 400

    image_data = request.json["image"].split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    img = cv2.resize(img, (200, 200))
    denoised_image = denoise_image(autoencoder, img)
    original_image = app.config.get('original_image')
    if original_image is None:
        return jsonify({"error": "Original image not found"}), 400
    denoised_snr = calculate_denoised_snr(original_image[..., 0], denoised_image[..., 0])
    denoised_base64 = image_to_base64(denoised_image)
    return jsonify({
        "denoised_image": denoised_base64,
        "denoised_snr": round(float(denoised_snr), 2) if denoised_snr != float('inf') else "∞"
    })

# Multiple image endpoints
@app.route("/upload_multiple", methods=["POST"])
def upload_multiple():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400
    file_names = []
    images = []
    for file in files:
        file_names.append(secure_filename(file.filename))
        img = load_image(file)
        images.append(img)
    app.config['original_images'] = images
    return jsonify({
        "count": len(files),
        "file_names": file_names
    })

@app.route("/denoise_multiple", methods=["POST"])
def denoise_multiple():
    original_images = app.config.get('original_images')
    if not original_images:
        return jsonify({"error": "No uploaded images found"}), 400
    output_images = []
    for img in original_images:
        denoised = denoise_image(autoencoder, img)
        snr = calculate_denoised_snr(img[..., 0], denoised[..., 0])
        composite = overlay_snr(denoised, round(float(snr), 2) if snr != float('inf') else "∞")
        output_images.append((composite, snr))
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (img, snr) in enumerate(output_images):
            _, buffer = cv2.imencode('.png', (img * 255).astype(np.uint8))
            filename = f"denoised_{idx+1}_SNR_{round(float(snr), 2) if snr != float('inf') else 'inf'}.png"
            zf.writestr(filename, buffer.tobytes())
    mem_zip.seek(0)
    zip_base64 = base64.b64encode(mem_zip.read()).decode('utf-8')
    zip_data = f"data:application/zip;base64,{zip_base64}"
    return jsonify({
        "zip_file": zip_data
    })

if __name__ == "__main__":
    app.run(debug=True)
