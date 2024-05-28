import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import os

def load_model():
    return MobileNetV2(weights='imagenet')

def classify_image(model, image):
    resized_image = image.resize((224, 224))
    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    return decoded_predictions

def annotate_image(image, predictions):
    draw = ImageDraw.Draw(image)
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()
    text_y = 10

    for i, (id, label, prob) in enumerate(predictions):
        text = f"{label} ({prob * 100:.2f}%)"
        draw.text((10, text_y), text, fill="red", font=font)
        text_y += font_size + 10

    return image

def process_image(model, image_path, save_path):
    image = Image.open(image_path)
    predictions = classify_image(model, image)
    annotated_image = annotate_image(image.copy(), predictions)
    annotated_image.save(save_path)
    return annotated_image

def process_video(model, video_path, save_folder):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(save_folder, 'output.avi'), cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        predictions = classify_image(model, image)
        annotated_image = annotate_image(image.copy(), predictions)

        cv2_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        out.write(cv2_image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if save_path:
            annotated_image = process_image(model, file_path, save_path)
            show_image(annotated_image)
        else:
            messagebox.showerror("Error", "Save path not provided.")
    else:
        messagebox.showerror("Error", "File not selected.")

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        save_folder = filedialog.askdirectory()
        if save_folder:
            process_video(model, file_path, save_folder)
            messagebox.showinfo("Info", f"Processed video saved to {save_folder}")
        else:
            messagebox.showerror("Error", "Save folder not selected.")
    else:
        messagebox.showerror("Error", "File not selected.")

def show_image(image):
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(root, image=image)
    panel.image = image
    panel.pack()

if __name__ == "__main__":
    model = load_model()

    root = tk.Tk()
    root.title("Image and Video Classifier")
    root.geometry("600x400")

    btn_upload_image = tk.Button(root, text="Upload Image", command=upload_image)
    btn_upload_image.pack(pady=20)

    btn_upload_video = tk.Button(root, text="Upload Video", command=upload_video)
    btn_upload_video.pack(pady=20)

    root.mainloop()
