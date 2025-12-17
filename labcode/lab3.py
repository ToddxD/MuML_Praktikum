# %% [markdown]
# # MuML Labor 3 - Machine Learning (Classic)
# 

# %% [markdown]
# ## 5.4.1 Teilaufgabe Merkmalsextraktion
# 
# Anwendung der Vorverarbeitungsschritte aus Labor 1 und der Merkmalsextraktion aus Labor 2.
# 
# Nachfolgend werden die Schritte aus Labor 1 und Labor 2 zusammengefasst. Um eine vergleichbare Ausgangssituation zu schaffen, werden die folgenden Merkmale extrahiert und in einem Pandas-Dataframe zurückgegeben und in der Tabelle ```features.csv``` gespeichert:
# 
# - img:  Index des Bildes ohne 000_OK.jpg, begonnen bei 0
# - file: Dateiname des Bildes
# - segment: fortlaufender Index des Segments,  beginnend bei 0
# - x, y, w, h: Koordinaten der Bounding Box des Segments
# - area: Fläche
# - hull_area: Fläche der konvexen Hülle
# - roundness: Rundheit
# - density: Dichte
# - eccentricity: Exzentrizität
# - orientation: Winkel in Grad im Bereich -90 ... +90 Grad
# - center_x, center_y: Schwerpunkt des Segments
# - mean: Mittelwert
# - std: Standardabweichung
# - class: Feld für Label, derzeit noch nicht bekannt

# %%
""" Feature extraction for solar cell images
    - preprocessing with shading correction
    - segmentation with fixed thresholding
    - contour finding
    - feature extraction
    - save features to a csv file

    @Author: Joerg Dahlkemper
    @Date: 2024-05-18
"""

import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import time
import timeit


IMAGE_PATH = "..\\labdata\\lab3_cells"         # path to the images
LIMITS = [70, 100, 900, 700]    # limits to eliminate black borders
BAR_HEIGHT = 50                 # height of the busbars
BAR_POSITIONS = [85, 395, 705]  # positions of the busbars
REF_IMG = "ref.jpg"             # reference image for shading correction
THRESH = 105                    # threshold for fixed thresholding 116
MAX_NUM_CNT = 3                 # maximum number of contours
MIN_AREA = 500                  # minimum area of a contour
RESULT_FILE = "features.csv"    # file to save the features


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
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) / (max_val - min_val) * 255
    stretched[stretched < 0] = 0
    stretched[stretched > 255] = 255
    return stretched.astype(np.uint8)

def smoothing(img, kernel_size):
    """ apply a Gauss filter to the image"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def shading_correction(img, ref_image):
    """ apply shading correction to the image"""
    img_filtered = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    ref_image_filtered = cv2.GaussianBlur(ref_image, (3, 3), 0).astype(np.float32)
    # shading correction
    img_shading_corrected = cv2.divide(img_filtered, ref_image_filtered, scale=128)
    # normalize the image so that median is 128
    median = np.median(img) - 128
    img_normalized = img_shading_corrected - median
    img_normalized[img_normalized < 0] = 0
    img_normalized[img_normalized > 255] = 255
    img_normalized = img_normalized.astype(np.uint8)
    # maximum filter to eliminate fingers
    img_normalized = cv2.dilate(img_normalized, np.ones((1, 5), np.uint8))
    return img_normalized

def preprocess(img):
    """ apply all preprocessing steps to the image"""
    img = get_roi(img, *LIMITS)
    img = discard_busbars(img, BAR_HEIGHT, BAR_POSITIONS)
    img = contrast_stretching(img)
    return img

def morph(img):
    """morph with diagonal lines"""
    strel1 = np.eye(5, 5, dtype=np.uint8)
    strel2 = strel1.copy()[::-1, :]
    strel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    img_morph = cv2.morphologyEx(img, cv2.MORPH_DILATE, strel1)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_DILATE, strel2)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, strel3)
    
    return img_morph

def segmentation(img, thresh):
    """segmentation of the image"""
    # fixed thresholding
    _, img_seg = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    # morph with diagonal lines
    img_seg = morph(img_seg)
    return img_seg

# contour finding
def find_contours(img, max_num_cnt, min_area):
    """ find contours in the image"""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by area
    contours_filtered = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    contours_sorted = sorted(contours_filtered, key=cv2.contourArea, reverse=True)[:max_num_cnt]
    return contours_sorted

def compute_features(contours, img_gray):
    """ contours_list contains a list of contours of a single image """
    features = []
    for contour in contours:

        # geometric features        
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        cirumference = cv2.arcLength(contour, True)
        roundness = 4 * np.pi * area / cirumference**2
        density = area / hull_area
        M = cv2.moments(contour)
        eccentricity = ((M['mu20'] - M['mu02'])**2 + 4 * M['mu11']**2 ) / (M['mu20'] + M['mu02'])**2
        orientation = np.rad2deg(0.5 * np.arctan2(2 * M['mu11'] , M['mu20'] - M['mu02']))
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])

        # compute mean and standard deviation of the contour
        mask = np.zeros(img_gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mean_val, stddev_val = cv2.meanStdDev(img_gray, mask=mask)
                        
        features.append([x, y, w, h, area, hull_area, roundness, density,
                         eccentricity, orientation, center_x, center_y,
                         mean_val[0][0], stddev_val[0][0]])
    return features    


def extract_features(path, ref_img, thresh, max_num_cnt, min_area, result_file):
    """ extract features from all images in a folder"""

    # create a sorted list for all images in the folder
    image_names = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") and filename != ref_img:
            image_names.append(filename)
    image_names.sort()

    # preprocess reference image
    ref_img = cv2.imread(os.path.join(path, ref_img), cv2.IMREAD_GRAYSCALE)
    ref_img = preprocess(ref_img)

    # iterate over all images
    extended_features = []  # features list for all images
    all_contours = []  # contours list for all images

    for img_idx, filename in enumerate(image_names):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        img = preprocess(img)
        img = shading_correction(img, ref_img)
        img_gray = img.copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = segmentation(img, thresh)
        contours = find_contours(img, max_num_cnt=max_num_cnt, min_area=min_area)
        features = compute_features(contours, img_gray)
        all_contours.append(contours)

        # extend features with image index and filename
        for k, feature in enumerate(features):
            extended_feature = [img_idx, filename, k, *feature, None]  # *feature unpacks the list
            extended_features.append(extended_feature)

        img_cnt = img_bgr.copy()

        for k, contour in enumerate(contours):
            cv2.drawContours(img_cnt, [contour], 0, (255, 0, 0), 2)
            center_x = features[k][10]
            center_y = features[k][11]
            cv2.putText(img_cnt, str(k), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # plot the images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_bgr)
        plt.title(filename)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_cnt)
        plt.title("Processed Image")
        plt.axis("off")
        plt.show()

    # generate feature table and save it to a csv file
    df = pd.DataFrame(extended_features, columns=['img', 'file', 'segment', 'x', 'y', 'w', 'h', 'area',
                                                  'hull_area', 'roundness', 'density',
                                                  'eccentricity', 'orientation', 'center_x',
                                                  'center_y', 'mean', 'std', 'class'])
    display(df)
    df.to_csv(os.path.join(path, result_file), index=False)
    return df, all_contours


if __name__ == "__main__":

    start_time = timeit.default_timer()
    result_file = "features.csv"
    df, contours = extract_features(IMAGE_PATH, REF_IMG, THRESH, MAX_NUM_CNT, MIN_AREA, RESULT_FILE)
    elapsed = timeit.default_timer() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Features saved to {os.path.join(IMAGE_PATH, result_file)}")
    print(f"Number of images: {df['file'].nunique()}")
    print(f"Number of segments: {len(df)}")


# %% [markdown]
# ### 5.4.1 Vergleich mit unserer Implementierung
# Unterschiede der Implementierung:
# - Bei der Shading Correction wurde in dem Beispielcode zuvor noch ein Gaussian Blur angewendet (Rauschreduzierung)
# - Beim Maximum Filter wurde im Beispielcode ein größerer Filternkern verwendet (1x5 statt 1x3)
# - Bei den morphologischen Operationen wurde im Beispielcode jeweils zwei Diagonale Filterkerne hintereinander verwendet, statt direkt ein Kreuz zu verwenden.
# - Bei den Features wurde im Beispielcode zusätzlich noch die konvexe Hülle, Standardabweichung und Durchschnitt ergänzt.
# - Zudem wurden im Beispielcode die Momente nur eimal berechnet, da dies eine aufwändige Operation ist.

# %% [markdown]
# ### 5.4.2 Labeling

# %%


for i, row in enumerate(df.iterrows()):
    img = cv2.imread('..\\labdata\\lab3_cells\\' + row[1]['file'], cv2.IMREAD_GRAYSCALE)
    img = preprocess(img)

    ref_img = cv2.imread(os.path.join('..\\labdata\\lab3_cells\\', REF_IMG), cv2.IMREAD_GRAYSCALE)
    ref_img = preprocess(ref_img)
    img = shading_correction(img, ref_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_index = row[1]['img']
    contour_index = row[1]['segment']
    
    cv2.drawContours(img_bgr, contours[img_index], contour_index, (255, 0, 0), 2)
    cv2.imshow('Image', img_bgr)
    key = cv2.waitKey(0)

    df.at[i, 'class'] = key - 48

    cv2.destroyAllWindows()

display(df)

# %% [markdown]
# ### Dateframe aus csv

# %%
df = pd.read_csv("Cell output 6 [DW].csv")

# %%
df.value_counts('class', sort=False)

# %%
# from pandas.plotting import scatter_matrix


# spalten = ['area', 'hull_area', 'roundness', 'density',
#            'eccentricity', 'orientation', 'mean', 'std', 'class']
# scatter_matrix(df[spalten], alpha=1, figsize=(20, 20), s= 70, c=df['class'], label=['Dark Area', 'Crack', 'Shunt', 'Finger'], marker='o', cmap='brg')

# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
 

spalten = ['area', 'hull_area', 'roundness', 'density',
           'eccentricity', 'orientation', 'mean', 'std', 'class']
 
sns.pairplot(
    df,
    vars=spalten,
    hue="class",
    palette="brg",
    plot_kws={"s": 70, "alpha": 0.7},
)
 
plt.show()

# %% [markdown]
# Klassen weglassen:
# - Orientation: alle Richtungen sind vertreten
# - Standardabweichung: ist in allen Kombinationen zu sehr gestreut
# - hull_area: ist in allen Kombinationen zu sehr gestreut, ist sehr ähnlich zur area
# - roundness: ist sehr ähnlich zur density
# 
# Klassen behalten:
# - area
# - mean
# - roundness
# - density
# 

# %%
import seaborn as sns
import matplotlib.pyplot as plt
 

spalten = ['area', 'roundness', 'density', 'mean', 'class']
 
sns.pairplot(
    df,
    vars=spalten,
    hue="class",
    palette="brg",
    plot_kws={"s": 70, "alpha": 0.7},
)
 
plt.show()

# %%
x = df[['area', 'roundness', 'density', 'mean']].values
y = df['class'].values

# Shuffling
np.random.seed(42)
rand = np.random.permutation(len(x))

x = x[rand]
y = y[rand]

# Splitting
split = int(len(x)*0.6)
x_train = x[:split]
y_train = y[:split]

x_val = x[split:]
y_val = y[split:]


# %%
def acc(y_pred, y, label):
    mask = (label == y) #alle label, die y entsprechen sind in der mask
    return np.mean(y_pred[mask] == y[mask])

def recall(y_pred, y, label):
    mask = (label == y)
    return sum(y[mask] & y_pred[mask]) / sum(y[mask])

def precision(y_pred, y, label):   
    return sum(y & y_pred == label) / sum(y_pred == label)

def predict_baseline(x):
    return np.full(len(x), 3) # immer Shunts

def confusion_matrix(y_pred, y):
    cm = np.zeros((4, 4), dtype=int)
    for i in range(len(y_pred)):
        cm[y[i]][y_pred[i]] += 1
    return cm

def overall_accuracy(y_pred, y):
    cm = confusion_matrix(y_pred, y)
    for i in range(4):
        sum += cm[i][i]

    return sum / len(y)
    

baseline_acc = acc(predict_baseline(x), y)

print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")


# %% [markdown]
# ### 5.4.6 k-Nearest Neighbour

# %%
k = 4

def predict_k_nearest(x):
    distances = []
    for i in range(x_train.shape[0]):
        distances.append(manhatten_distance(x_train[i], x, y_train[i]))

    distances.sort()
    neighbors = distances[:k]
    return max(set(neighbors), key=neighbors.count)[1]

def manhatten_distance(punkt_1, punkt_2, label):
    return (np.sum(np.abs(punkt_1 - punkt_2)), label)


predictions = []
for i in range(len(x_val)):
    predictions.append(predict_k_nearest(x_val[i]))

predictions = np.array(predictions)
print("Accuracy Dark Area: ", acc(predictions, y_val, 1))
print("Accuracy Crack: ", acc(predictions, y_val, 2))
print("Accuracy Shunt: ", acc(predictions, y_val, 3))
print("Accuracy Finger: ", acc(predictions, y_val, 4))

print("Recall Dark Area: ", recall(predictions, y_val, 1))
print("Recall Crack: ", recall(predictions, y_val, 2))
print("Recall Shunt: ", recall(predictions, y_val, 3)) 
print("Recall Finger: ", recall(predictions, y_val, 4))

print("Precision Dark Area: ", precision(predictions, y_val, 1))
print("Precision Crack: ", precision(predictions, y_val, 2))
print("Precision Shunt: ", precision(predictions, y_val, 3))
print("Precision Finger: ", precision(predictions, y_val, 4))

print("Overall Accuracy: ", overall_accuracy(predictions, y_val))

print("Confusion Matrix: \n", confusion_matrix(predictions, y_val))



# %% [markdown]
### 5.4.7 Teilaufgabe Support Vector Machine (SVM)
# Es wird ein nichtlinearer Kernel verwendet, da dadurch eine lineare Trennung durch Ebenen ermöglicht wird.
# %%
from sklearn.svm import SVC

#svm = cv2.ml.SVM_create() #openCV
#TODO: degree 2,3,4 und c 10, 100, 1000(kosten) anpassen
svm = SVC(kernel='poly', C=1, degree=3, gamma=1)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_val)


# %% [markdown]
### 5.4.8 Teilaufgabe Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
def decision_tree():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x)

