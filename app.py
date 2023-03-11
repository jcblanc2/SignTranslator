import tkinter as tk
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

offset = 20
imgSize = 300

counter = 0


class App:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Sign Translator")
        self.window.iconbitmap('images/icon.ico')
        self.window.configure(bg="#151226")
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                        'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        self.model = tf.keras.models.load_model('saved_model/model.h5')
        self.filepath = ""
        self.predictions = []

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.detector = HandDetector(maxHands=1)
        hands, img = self.detector.findHands(self.vid)

        # if hands:
        #     hand = hands[0]
        #     x, y, w, h = hand['bbox']
        #
        #     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        #     imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        #
        #     imgCropShape = imgCrop.shape
        #
        #     aspectRatio = h / w
        #
        #     if aspectRatio > 1:
        #         k = imgSize / h
        #         wCal = math.ceil(k * w)
        #         imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        #         imgResizeShape = imgResize.shape
        #         wGap = math.ceil((imgSize - wCal) / 2)
        #         imgWhite[:, wGap:wCal + wGap] = imgResize
        #         # prediction, index = classifier.getPrediction(imgWhite, draw=False)
        #         # print(prediction, index)

        # else:
        #     k = imgSize / w
        #     hCal = math.ceil(k * h)
        #     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        #     imgResizeShape = imgResize.shape
        #     hGap = math.ceil((imgSize - hCal) / 2)
        #     imgWhite[hGap:hCal + hGap, :] = imgResize
        # prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT), bg="#151226")
        self.canvas.pack()

        # Entry widget for adding text to the canvas
        self.text_entry = tk.Entry(window, width=50, highlightbackground="#151226")
        self.text_entry.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tk.Button(window, text="Snapshot", bg="#bf307f", width=15, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Three buttons for adding, deleting, and clearing the canvas
        self.btn_add = tk.Button(window, text="Add", width=10, bg="#bf307f", command=self.add_canvas_item)
        self.btn_add.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_delete = tk.Button(window, text="Delete", width=10, bg="#bf307f", command=self.delete_canvas_item)
        self.btn_delete.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_clear = tk.Button(window, text="Clear", width=10, bg="#bf307f", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.LEFT, padx=5, pady=5)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def prepare_image(self):
        img = cv2.imread(self.filepath)
        image = cv2.resize(img, (64, 64))
        img_array = np.expand_dims(image, axis=0)
        reshaped_image = np.reshape(img_array, (-1, 64, 64, 3))
        return reshaped_image

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Convert the frame from OpenCV to PIL format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # Add the image to the canvas
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def add_canvas_item(self):
        # Add text to the canvas
        text = self.text_entry.get()
        x = self.canvas.winfo_width() / 2
        y = self.canvas.winfo_height() / 2
        self.canvas.create_text(x, y, text=text, fill='white', font=('Arial', 24))

    def delete_canvas_item(self):
        # Delete the last item added to the canvas
        item_list = self.canvas.find_all()
        if item_list:
            last_item = item_list[-1]
            self.canvas.delete(last_item)

    def clear_canvas(self):
        # Clear all items from the canvas
        self.canvas.delete("all")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Convert the frame from OpenCV to PIL format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # Add the image to the canvas
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


# Create the main window and start the app
root = tk.Tk()
app = App(root)
