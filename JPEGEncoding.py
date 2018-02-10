import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

B=8 # blocksize (In Jpeg the

cv2.CV_LOAD_IMAGE_UNCHANGED=1

img1 = cv2.imread("images/lena.bmp", cv2.CV_LOAD_IMAGE_UNCHANGED)
h,w=np.array(img1.shape[:2])/B * B
img1=img1[:h,:w]

#Convert BGR to RGB
img2=np.zeros(img1.shape,np.uint8)
img2[:,:,0]=img1[:,:,2]
img2[:,:,1]=img1[:,:,1]
img2[:,:,2]=img1[:,:,0]
plt.imshow(img2)

point=plt.ginput(1)
block=np.floor(np.array(point)/B) #first component is col, second component is row
print "Coordinates of selected block: ",block
scol=block[0,0]
srow=block[0,1]
plt.plot([B*scol,B*scol+B,B*scol+B,B*scol,B*scol],[B*srow,B*srow,B*srow+B,B*srow+B,B*srow])
plt.axis([0,w,h,0])

transcol=cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)

SSV=2
SSH=2
crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(1,1))
cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(1,1))

crsub=crf[::SSV,::SSH]
cbsub=cbf[::SSV,::SSH]
imSub=[transcol[:,:,0],crsub,cbsub]