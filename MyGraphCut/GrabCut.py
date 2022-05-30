import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data/DSC00002.JPG')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (400,50,650,img.shape[1]-50)

cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv.imwrite('grabcut.JPG', img)
#plt.imshow(img),plt.colorbar(),plt.show()