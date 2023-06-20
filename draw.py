import pygame
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

black = (0,0,0)
white = (255,255,255)
window_width = 560
window_height = 560
rows = 28
cols = 28
blocksize = 20

def draw_grid(window):

    window.fill(white)

    for x in range(window_width):
        for y in range(window_height):
            pygame.draw.rect(window,white,(x*blocksize,
                        y*blocksize,
                        blocksize,
                        blocksize),0)
    make_grid_array()

def make_grid_array():
    global array_2d
    array_2d = [[0. for x in range(rows)] for x in range(cols)]

def make_black_grid(x,y):

    global array_2d

    pygame.draw.rect(window,black,(x*blocksize,
            y*blocksize,
            blocksize,
            blocksize),0)
    pygame.display.update()

    array_2d[y][x] = 1.

def get_posit():
    if pygame.mouse.get_pressed()[0]:
            try:
                position = pygame.mouse.get_pos()
                x,y = position
                x = x//blocksize
                y = y//blocksize
                make_black_grid(x,y)
            except AttributeError:
                pass

def Guess(numpy_arr):

    a = np.array(array_2d,dtype=float)
    a = np.reshape(a,[1,28,28,1])

    model = keras.models.load_model("model_mnist_num.h5")

    prediction = model.predict(a)
    ans = np.argmax(prediction[0])
    print(ans)
    return ans

def main():
    global array_2d

    run = True
    drawrun = True
    while run:
        
        for event in pygame.event.get():
            if drawrun:
                get_posit()
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    drawrun=False
                    array_to_guess = np.array(array_2d)
                    ret = Guess(array_to_guess)
                    break
    pygame.display.update()

pygame.init()
window = pygame.display.set_mode((window_width,window_height))
pygame.display.set_caption("Guess The Number")
draw_grid(window)
pygame.display.update()
main()
