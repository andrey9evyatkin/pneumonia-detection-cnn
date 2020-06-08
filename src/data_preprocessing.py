import pandas as pd
import cv2
import numpy as np


def get_dataframe(data_set):
    normal_cases = mark_data(data_set[0], 0)
    pneumonia_cases = mark_data(data_set[1], 1)
    data = normal_cases[0] + pneumonia_cases[0]
    labels = normal_cases[1] + pneumonia_cases[1]
    return pd.DataFrame(zip(data, labels), columns=['image', 'class'], index=None)


def mark_data(data, label):
    result = []
    labels = []
    for d in data:
        image = normalize_image(d)
        result.append(image)
        labels.append(label)
    return result, labels


def normalize_image(image):
    img = cv2.imread(str(image))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    return img
