from microbit import *
import random


images = [
    Image.HEART,
    Image.HAPPY,
    Image.SMILE,
    Image.SAD
]

while True:
    for img in images:
        display.show(img)
        sleep(3000)  
