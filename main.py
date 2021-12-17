from yolox import getModel as yoloxModel
#from yolov5 import getModel as yolov5Model
#from openVino import getModel as openvinoModel
import cv2,random,os
from ed_utils import image_resize,vis,COCO_CLASSES
import numpy as np
from time import time
import traceback

classes = COCO_CLASSES

device = 'cpu'

maxsize = 800
cls_conf=0.1


model = yoloxModel("yourModel",(640,640),classes,device)
#model = yolov5Model("yolov5/pretrained/kedaireka.pt",(640,640),classes,'cpu')
#model = openvinoModel("openVino/pretrained/kedaireka-d.xml",(640,640),classes,'CPU')

kamera = cv2.VideoCapture(0)

while 1:
    mulai = time()
    _,g = kamera.read()

    if _:
       
        if g.shape[0] >= g.shape[1]:
            g = image_resize(g,height=maxsize)
        elif g.shape[0] < g.shape[1]:
            g = image_resize(g,width=maxsize)

        result = model.inference(g.copy())
        print(result)

        if result is not None:
            bboxes, scores, cls = result
            bboxes, scores, cls = bboxes.tolist(), scores.tolist(), cls.tolist() 
    
            
            g = vis(g, bboxes, scores, cls, cls_conf, classes)

        fps = f'fps:{round(1/(time()-mulai),2)}'
        g = cv2.putText(g,fps,(50,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
        cv2.imshow("aa",g)
        
 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()




