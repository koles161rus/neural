import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL
from tkinter import *

width = 128
height = 128
center = height//2

def save():
    filename = 'number.png'
    image1.save(filename)

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill='black', width=5)
    draw.line([x1, y1, x2, y2], fill='black', width=5)

root = Tk()

cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

image1 = PIL.Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image1)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

def button_functions(save, exit):
    save = save()
    exit = root.destroy()

button=Button(text="Принять", command=lambda: button_functions(save, exit))
button.pack()
root.mainloop()

json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mnist_model.h5")

loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

img_path = 'number.png'
img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
plt.imshow(img, cmap='gray')

x = image.img_to_array(img)
x = 255 - x
x /= 255
x = np.expand_dims(x, axis=0)

prediction = loaded_model.predict(x)
print(np.argmax(prediction, axis=1))