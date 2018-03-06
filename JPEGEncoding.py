#https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from Rle import *
from PIL import Image
from scipy import interpolate


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

# point=plt.ginput(1)
# block=np.floor(np.array(point)/B) #first component is col, second component is row
# print "Coordinates of selected block: ",block
# scol=block[0,0]
# srow=block[0,1]
# plt.plot([B*scol,B*scol+B,B*scol+B,B*scol,B*scol],[B*srow,B*srow,B*srow+B,B*srow+B,B*srow])
# plt.axis([0,w,h,0])

transcol=cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
# cv2.COLOR_YCrCb2RGB

SSV=2
SSH=2
crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(1,1))
cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(1,1))

crsub=crf[::SSV,::SSH]
cbsub=cbf[::SSV,::SSH]
#imSub=[transcol[:,:,0],crsub,cbsub]
imSub= [transcol[:,:,0]]

QY=np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,48,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

# QC=np.array([[17,18,24,47,99,99,99,99],
#                          [18,21,26,66,99,99,99,99],
#                          [24,26,56,99,99,99,99,99],
#                          [47,66,99,99,99,99,99,99],
#                          [99,99,99,99,99,99,99,99],
#                          [99,99,99,99,99,99,99,99],
#                          [99,99,99,99,99,99,99,99],
#                          [99,99,99,99,99,99,99,99]])
TransAll=[]
TransAll_I=[]
TransAllQuant=[]
TransAllDequant=[]
RLE_Trans_Quant = []
ch=['Y','Cr','Cb']
plt.figure()

# QF=99.0
# if QF < 50 and QF > 1:
#         scale = np.floor(5000/QF)
# elif QF < 100:
#         scale = 200-2*QF
# else:
#         print "Quality Factor must be in the range [1..99]"
# scale=scale/100.0
# Q=[QY*scale,QC*scale,QC*scale]

zigZag_Matrice_all = []
DC_Values = []
huffman_symbole_codes = []
for idx,channel in enumerate(imSub):
        plt.subplot(1,3,idx+1)
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        Trans = np.zeros((channelrows,channelcols), np.float32)
        Trans_I = np.zeros((channelrows,channelcols), np.float32)
        TransQuant = np.zeros((channelrows,channelcols), np.float32)
        TransDequant = np.zeros((channelrows,channelcols), np.float32)
        RLE_Trans_Quant =  np.zeros((channelrows,channelcols), np.float32)
        blocksV=channelrows/B
        blocksH=channelcols/B
        vis0 = np.zeros((channelrows,channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis1 = np.zeros((channelrows,channelcols), np.float32)
        vis1[:channelrows, :channelcols] = channel
        vis0=vis0-128

        for row in range(blocksV):
                for col in range(blocksH):
                        currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                        Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                        TransQuant[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/QY)


                        TransDequant[row*B:(row+1)*B,col*B:(col+1)*B]=\
                            np.round(TransQuant[row*B:(row+1)*B,col*B:(col+1)*B]*QY)

                        # print currentblock
                        # print TransDequant[row*B:(row+1)*B,col*B:(col+1)*B]

                        currentblock2=cv2.idct(TransDequant[row*B:(row+1)*B,col*B:(col+1)*B])
                        # print TransDequant
                        # print TransQuant
                        # print currentblock2
                        # print vis0[row*B:(row+1)*B,col*B:(col+1)*B]
                        Trans_I[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock2

                        zigZag_Matrice = getZigZag(TransQuant[row*B:(row+1)*B,col*B:(col+1)*B])
                        DC_Values.append(zigZag_Matrice[0])
                        zigZag_Matrice_all.append(zigZag_Matrice[1:])
                        huffman_symbole_codes.append(dict(huffman(zigZag_Matrice[1:])))
        # DPCM(DC_Values)
        encoded_image = rle(zigZag_Matrice_all, huffman_symbole_codes)
        # print encoded_image
        TransAll.append(Trans)
        Trans_I = Trans_I+128
        TransAll_I.append(Trans_I)
        TransAllQuant.append(TransQuant)
        TransAllDequant.append(TransDequant)


# print Trans_I
# outputBlock_RGB = cv2.cvtColor(Trans_I, cv2.COLOR_YCrCb2RGB)

concatenated_list = np.concatenate( TransAll_I, axis=0 )
# outputBlock_RGB = cv2.cvtColor( ,cv2.COLOR_YCrCb2RGB)
# outputBlock_RGB = cv2.cvtColor(concatenated_list, cv2.COLOR_YCrCb2RGB)
# fft_p = abs(np.fft.rfft2(concatenated_list))
# im = Image.fromarray(fft_p)
# im = im.convert('RGB')
# print Trans_I
# print transcol
# print TransAll_I
# im.save("images/your_file.bmp")
# img2 = cv2.imread("images/your_file.bmp", cv2.CV_LOAD_IMAGE_UNCHANGED)
# outputBlock_RGB = cv2.cvtColor(img2, cv2.COLOR_YCrCb2RGB)
# outputBlock_RGB = cv2.cvtColor(im, cv2.COLOR_YCrCb2RGB)

print len(transcol)
print len(concatenated_list)

x = np.array(range(512))
y = np.array(range(512))
a = concatenated_list[:]
xx, yy = np.meshgrid(x, y)
f = interpolate.interp2d(x, y, a, kind='linear')

znew = f(x, y)

print znew
print TransAll_I

        # if idx==0:
        #         selectedTrans=Trans[int(srow*B):(int(srow+1)*B),int(scol*B):int((scol+1)*B)]
        # else:
        #         sr=np.floor(srow/SSV)
        #         sc=np.floor(scol/SSV)
        #         selectedTrans=Trans[sr*B:(sr+1)*B,sc*B:(sc+1)*B].3
        
        #plt.imshow(selectedTrans,cmap=cm.jet,interpolation='nearest')
        #plt.colorbar(shrink=0.5)
        #plt.title('DCT of '+ch[idx])
        #plt.show()
