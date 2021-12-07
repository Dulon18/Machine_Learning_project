import cv2
import numpy as np
from PIL import Image


def PIX(event,x,y,flags,param):
    if event ==cv2.EVENT_LBUTTONDBLCLK:
        r,g,b = rgb_img.getpixel((x,y))
        txt = str(r)+","+str(g)+","+str(b)
        bg = np.zeros((200,400,3),np.uint8)
        bg[:,0:400]=(b,g,r)
        font =cv2.FONT_ITALIC
        cv2.putText(bg,txt,(10,100),font,1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('rgb color #',bg)


cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    flipped = cv2.flip(frame,1)
    cv2.imshow('vid',flipped)
    if cv2.waitKey(1) & 0xFF == ord('c'): #press c for take picture
        cv2.imwrite('s.jpg',flipped)
        image = Image.open('s.jpg')
        rgb_img = image.convert('RGB')
        cv2.imshow('picture',flipped)
        cv2.setMouseCallback('picture',PIX)
    elif cv2.waitKey(1) & 0xFF == ord(' '): #hit space to quit
        break

cap.release()
cv2.destroyAllWindows()

