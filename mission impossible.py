import cv2 as cv 
import  numpy as np 
import os 
from matplotlib import pyplot as plt
i = 0 # for saving image 
Match_Count = 10



SavePath= 'C:/Users/EERD_S54_SKU1/Desktop/mission impossible/image'
cap = cv.VideoCapture(0)

if not cap.isOpened():
 print("Cannot open camera")
 exit()

def saveimage(filename, final, SavePath):
    global i 
    if cv.waitKey(1) == ord('s'):
        save_filename = os.path.join(SavePath, f"{filename}{i}.png")
        if not os.path.exists(save_filename):
            cv.imwrite(save_filename, final)
            print(f'Saved {save_filename}')
        i += 1

 
def deal(img): #image processing & Conditional judgment
   gaus = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
   finalimg = cv.cvtColor(gaus,cv.COLOR_RGB2BGR)
   return finalimg



while True:
   ret,frame = cap.read()
   final = deal(frame)

   cv.imshow("final",final)

   saveimage("image",final,SavePath)


   if cv.waitKey(1) == ord('q'):
    break