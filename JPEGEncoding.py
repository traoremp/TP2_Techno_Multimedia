#https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from Rle import * 

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

SSV=2
SSH=2
crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(1,1))
cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(1,1))

crsub=crf[::SSV,::SSH]
cbsub=cbf[::SSV,::SSH]
imSub=[transcol[:,:,0],crsub,cbsub]
#imSub= [transcol[:,:,0]]

QY=np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,48,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

QC=np.array([[17,18,24,47,99,99,99,99],
                         [18,21,26,66,99,99,99,99],
                         [24,26,56,99,99,99,99,99],
                         [47,66,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99]])
TransAll=[]
TransAllQuant=[]
RLE_Trans_Quant = []
ch=['Y','Cr','Cb']
plt.figure()

QF=90.0
if QF < 50 and QF > 1:
        scale = np.floor(5000/QF)
elif QF < 100:
        scale = 200-2*QF
else:
        print "Quality Factor must be in the range [1..99]"
scale=scale/100.0
Q=[QY*scale,QC*scale,QC*scale]


compressed = {}
for idx,channel in enumerate(imSub):
#plt.subplot(1,3,idx+1)
        zigZag_Matrice_all = []
        DC_Values = []
        huffman_symbole_codes = []
        vector_matrices = []
        encoded_DC_Values = '' 

        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        Trans = np.zeros((channelrows,channelcols), np.float32)
        TransQuant = np.zeros((channelrows,channelcols), np.float32)
        blocksV=channelrows/B
        blocksH=channelcols/B
        vis0 = np.zeros((channelrows,channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis0=vis0-128

        for row in range(blocksV):
                for col in range(blocksH):
                        currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                        Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                        TransQuant[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/Q[idx])
                        zigZag_Matrice = getZigZag(TransQuant[row*B:(row+1)*B,col*B:(col+1)*B])
                        DC_Values.append(zigZag_Matrice[0])
                        vector_matrices = vector_matrices + zigZag_Matrice[1:]
                        zigZag_Matrice_all.append(zigZag_Matrice[1:])
        huffman_symbole_codes = dict(huffman(vector_matrices))
        encoded_DC_Values = DPCM(DC_Values)
        encoded_image = rle(zigZag_Matrice_all, huffman_symbole_codes)
        TransAll.append(Trans)
        TransAllQuant.append(TransQuant)
        compressed[idx] = compress(encoded_DC_Values, encoded_image, huffman_symbole_codes)
compressed_size = 0
for idx, compressed_channel in compressed.items():
        compressed_size += len(compressed_channel)
print h * w * 8 * 3
print compressed_size
print "Compression ratio = %f"%(1-(float(compressed_size)/float(h * w * 8 * 3)))
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