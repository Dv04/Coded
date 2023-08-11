from flask import Flask, send_from_directory
import os
import random

app = Flask(__name__)
image_folder = "/Users/apple/Coded/Flask/Frames"


@app.route("/")
def get_image():
    images = os.listdir(image_folder)
    image_path = random.choice(images)
    return send_from_directory(image_folder, image_path)


@app.route("/next_image")
def next_image():
    # You can add logic here to change the image
    return {"status": "success"}


if __name__ == "__main__":
    app.run()
