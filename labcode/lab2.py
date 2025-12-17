# %% [markdown]
# # MuML Labor 2 - Segmentierung und Merkmalsextraktion
# 

# %% [markdown]
# ## 4.4.1 Teilaufgabe Vorverarbeitung
# 
# Anwendung der Vorverarbeitungsschritte aus Labor 1 auf jedes der Bilder im Ordner images
# 1. get_roi: Berandung entfernen
# 2. discarc_busbars: Busbars entfernen
# 3. contrast_stretching: Grauwertspreizung auf 0 ... 255
# 4. smoothing: Glättung mit Gauss-Kernel der Größe 5
# 
# Die zugehörigen Dateinamen werden in ```image_names``` gespeichert, die Bilder zunächst in einer Liste ```image_list``` und anschließend angezeigt und in ein numpy array ```images``` überführt.

# %%
""" Preprocessing of the images for Lab 2
    - eliminate black borders and diagonal edges
    - discard busbars
    - apply contrast stretching
    - apply smoothing

    input: images in the folder images
    output: images after preprocessing as numpy array

    @Author: Joerg Dahlkemper
    @Date: 2024-04-21
"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import timeit

IMAGE_PATH = "..\\labdata\\lab1u2_images"
LIMITS = [70, 50, 900, 900]
BAR_HEIGHT = 45
BAR_POSITIONS = [85, 395, 705]


def get_roi(img, row1, col1, row2, col2):
    """ eliminate black borders and diagonal edges from the image"""
    return img[row1:row2, col1:col2]


def discard_busbars(img, bar_height, positions):
    """ discard busbars from the image"""
    height, width = img.shape
    count_bars = len(positions)

    img_discarded_bars = img.copy()

    # apply numpy delete in a loop to remove the busbars
    for step, position in enumerate(positions):
        img_discarded_bars = np.delete(img_discarded_bars, 
            np.arange(position - step * bar_height, position + (1 - step) * bar_height), axis=0)

    return img_discarded_bars


def contrast_stretching(img):
    """ apply contrast stretching to the image"""
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) / (max_val - min_val) * 255
    return stretched.astype(np.uint8)

def smoothing(img, kernel_size):
    """ apply a Gauss filter to the image"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# iterate over all images in the folder
image_names = []

for filename in os.listdir(IMAGE_PATH):
    if filename.endswith(".jpg") and filename.startswith("00"):
        image_names.append(filename)
image_names.sort()

image_list = []

for filename in image_names:
    print("Processing image: ", filename)
    img = cv2.imread(os.path.join(IMAGE_PATH, filename), cv2.IMREAD_GRAYSCALE)
    img = get_roi(img, *LIMITS)
    img = discard_busbars(img, BAR_HEIGHT, BAR_POSITIONS)
    img = contrast_stretching(img)
    # img = smoothing(img, 5)
    image_list.append(img)

images = np.array(image_list)
print("Dimensions of array images:", images.shape)

def plot_images(images, image_names, suptitle):
    # plot the images, two per row with the image name as title and a super title
    fig = plt.figure(figsize=(12, 24))

    for i, image_name in enumerate(image_names):
        plt.subplot(len(image_names)//2 + 1, 2, i+1)
        plt.imshow(images[i], vmin=0, vmax=255, cmap='gray')
        plt.title(image_name)
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=16, y=1.02)
    plt.show()

plot_images(images[1:], image_names[1:], "Preprocessed images")


# %% [markdown]
# ## 4.4.2 Teilaufgabe Dark Areas
#
# Es ist ein Algorithmus zu entwickeln, der die flächigen dunklen Bereiche in den Solar-
# zellenbildern erkennt und visualisiert. Hierzu zählen insbesondere Disconnected Areas,
# Printing Defects und Fingerfehler.
# Ausgehend von den gemäß 4.4.1 bereitgestellten Bildern sind die folgenden Ansätze der
# schwellwertbasierten Segmentierung zu verfolgen und deren Eignung jeweils als Markdown-
# Zelle zu bewerten:
# 1. automatische Schwellwertbestimmung mittels Otsu-Algorithmus
# 2. manuell gewählte Schwelle wobei als Referenzbild das Bild 000_... zu verwenden
# ist.
# Die Ergebnisse jedes Teilversuchs sind zu visualisieren und qualitativ auf Eignung zu
# bewerten.

# %%
def otsu(image) :
    thresh_otsu, image_otsu = cv2.threshold(cv2.bitwise_not(image.astype(np.uint8)), 0, 255, cv2.THRESH_OTSU)
    print("Otsu: %d" % thresh_otsu)
    return image_otsu

# manuelle Schwellwertbestimmung
def manuelle_schwellwertbestimmung(image, thresh) :
    t, image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)
    return image


otsu_imgs = []
manuelle_imgs = []

for image in images[1:]:
    
    otsu_imgs.append(otsu(image))
    manuelle_imgs.append(manuelle_schwellwertbestimmung(image, 75))

plot_images(otsu_imgs, image_names[1:], "Otsu images")
plot_images(manuelle_imgs, image_names[1:], "Manueller Schwellwert images")



    

# %% [markdown]
# ## 4.4.3 Teilaufgabe Shading Correction
# Die Leistungsfähigkeit der Segmentierung ist aufgrund der Helligkeitsunterschiede be-
# grenzt und soll durch eine Shading Correction wie folgt verbessert werden:
# 1. Shading Correction mittels Division durch das Referenzbils 000_...
# 2. Additive Grauwertverschiebung, so dass der Median jedes Bildes 128 beträgt
# 3. Anwendung der Schwellwertsegmentierung auf das korrigierte Bild
# Die genannten Schritte sind zu implementieren und je Schritt die Ergebnisse zu visuali-
# sieren und zu bewerten. Es ist zu erklären, welchen Effekt die Grauwertverschiebung auf
# die Segmentierung hat.

# %%
def division(image1, image2) :
    return cv2.divide(image1, image2)

div_imgs = []
shifted_imgs = []


for image in images[1:]:
    div_img = division(image, images[0])
    div_imgs.append(div_img)
    median_of_img = np.median(div_img)
    offset = 128.0 - median_of_img

    median_img = div_img + offset.astype(np.uint8)

    median_img[median_img < 0] = 0
    median_img[median_img > 255] = 255
    median_img = median_img.astype(np.uint8)
    
    shifted_imgs.append(median_img)

# %%
plot_images(div_imgs, image_names[1:], "Shading corrected (division) images")
# %%
plot_images(shifted_imgs, image_names[1:], "Grauwertverschoben images")

# %%
div_imgs_otsu = []
shifted_imgs_manuell = []
for image in shifted_imgs:
    div_imgs_otsu.append(otsu(image))
    shifted_imgs_manuell.append(manuelle_schwellwertbestimmung(image, 127))

    
# %%
plot_images(div_imgs_otsu, image_names[1:], "shading corrected, grauwertverschoben, otsu images")
# %%
plot_images(shifted_imgs_manuell, image_names[1:], "shading corrected, grauwertverschoben, manuelle Schwellwert images")

    
# %% [markdown]
# ## 4.4.4 Teilaufgabe Rangordnungsfilter
# Eine hinreichend empfindliche Segmentierung von Defekten führt auch zu einer Seg-
# mentierung von Fingern. Da sich Finger dunkel abheben, ist ein Maximum- Filter für
# das Grauwertbild nach der Shading Correction und Grauwertverschiebung gemäß 4.4.3
# vor Anwendung der Schwellwertsegmentierung zu implementieren. Dieser soll die feh-
# lerhafte Segmentierung der Finger unterdrücken, ohne die Detektion von Defekten zu
# beeinträchtigen. Das strukturierende Element soll dabei die Form 1 x 3 Pixel haben. Ein
# Maximum-Filter ist in der OpenCV-Bibliothek über die auf einem Grauwertbild ange-
# wandte Funktion cv2.dilate() realisierbar.
# Die Ergebnisse der optimierten Segmentierung sind zu visualisieren und zu bewerten

# %%
img_dilated = []

for image in shifted_imgs:
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    img_dilated.append(cv2.dilate(image, struct))

plot_images(img_dilated, image_names[1:], "dilated images")

# %%
img_dilated_otsu = []
img_dilated_manuell = []
for image in img_dilated:
    img_dilated_otsu.append(otsu(image))
    img_dilated_manuell.append(manuelle_schwellwertbestimmung(image, 127))

plot_images(img_dilated_otsu, image_names[1:], "dilated + otsu images")
plot_images(img_dilated_manuell, image_names[1:], "dilated + manuell images")

