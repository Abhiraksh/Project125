# from sklearn.datasets import fetch_openml
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split as tts
# from sklearn.linear_model import LogisticRegression as lr
# from PIL import Image
# import PIL.ImageOps 

# x,y = fetch_openml("mnist_784", version = 1, return_X_y = True)
# xTrain, xTest, yTrain, yTest = tts(x, y, test_size = 2500, train_size = 7500)

# xTrainSc = xTrain/255.0
# xTestSc = xTest/255.0

# LR = lr(solver = "saga", multi_class = "multinomial").fit(xTrainSc, yTrain)

# def getPrediction(image):
#     # im_pil = Image.open(image)
#     # imgBW = im_pil.convert("L")
#     # imgResized = imgBW.resize((28,28), Image.ANTIALIAS)
#     # pixelFilter = 20
#     # min_pixel = np.percentile(imgResized, pixelFilter)
#     # imgScaled = np.clip(imgResized - min_pixel, 0, 255)
#     # max_pixel = np.max(imgResized)
#     # imgScaled = np.asarray(imgScaled)/max_pixel
#     # testSample = np.array(imgScaled).reshape(1,784)
#     # testPrediction = LR.predict(testSample)
#     # return testPrediction[0]
#     im_pil = Image.open(image)
#     image_bw = im_pil.convert('L')
#     image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
#     pixel_filter = 20
#     min_pixel = np.percentile(image_bw_resized, pixel_filter)
#     image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
#     max_pixel = np.max(image_bw_resized)
#     image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
#     test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
#     test_pred = LR.predict(test_sample)
#     return test_pred[0]
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]

