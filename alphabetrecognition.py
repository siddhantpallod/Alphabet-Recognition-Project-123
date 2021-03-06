import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
classesLen = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state = 9, train_size = 3500, test_size = 500)
xTrainScaled = xTrain / 255
xTestScaled = xTest / 255

lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial')
cl = lr.fit(xTrainScaled, yTrain)

yPredict = cl.predict(xTestScaled)
accuracy = accuracy_score(yTest, yPredict)
print('Accuracy is: ', accuracy)

video = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width / 2) - 56, int(height / 2 - 56))
        bottomRight = (int(width / 2) + 56, int(height / 2) + 56)
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]
        imagePil = Image.fromarray(roi)
        imageBw = imagePil.convert('L')
        imageBwResize = imageBw.resize((28, 28), Image.ANTIALIAS)
        imageBwResizeInverted = PIL.ImageOps.invert(imageBwResize)
        pixelFiler = 20
        minPixel = np.percentile(imageBwResizeInverted, pixelFiler)
        imageBwResizeInvertedScaled = np.clip(imageBwResizeInverted - minPixel, 0, 255)
        maxPixel = np.max(imageBwResizeInverted)
        imageBwResizeInvertedScaled = np.asarray(imageBwResizeInvertedScaled) / maxPixel
        testSample = np.array(imageBwResizeInvertedScaled).reshape(1, 784)
        testPredict = cl.predict(testSample)
        print("Predicted class is: " , testPredict)

        if cv2.waitKey(1) & 0xFF == ord('Q')  :
            break
    
    except Exception as e:
        pass

video.release()
cv2.destroyAllWindows()

