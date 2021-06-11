import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=0, train_size=7500, test_size=2500)
xtrainscale=xtrain/255
xtestscale=xtest/255
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscale,ytrain)
ypred=clf.predict(xtestscale)
print('accuracy',accuracy_score(ytest,ypred))

cap=cv2.VideoCapture(0)
while(True):
    try:
        ret, frame=cap.read()
        height, weight=gray.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft, bottomright, (0,255,0),2)
        ROI=gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        imPil=Image.fromarray(ROI)
        imgbw=imPil.convert('L')
        imgbwresize=imgbw.resize((28,28),Image.ANTIALIAS)
        imgbwresizeinv=PIL.ImageOps.invert(imgbwresize)
        pixelfilter=20
        minpixel=np.percentile(imgbwresizeinv,pixelfilter)
        imgbwresizeinvscale=np.clip(imgbwresizeinv-minpixel,0,255)
        maxpixel=np.max(imgbwresizeinv)
        imgbwresizeinvscale=np.asarray(imgbwresizeinvscale)/maxpixel
        testsample=np.array(imgbwresizeinvscale).reshape(1,784)
        testpred=clf.predict(testsample)
        print('predicted class is',testpred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)& 0xFF==ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()