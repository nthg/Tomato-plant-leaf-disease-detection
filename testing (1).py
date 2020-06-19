import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import random
#path = 'E:\\khk\\volumeF\\soft1\\WinPython\\settings\\.spyder-py3\\cylinre\\test'
path = 'E:\\khk\\volumeF\\soft1\\WinPython\\settings\\.spyder-py3\\pdetector\\0'
path = 'C:\\Users\\Hp\Desktop\\drive-download-20191216T11352Z-001\\test\\'
disease=['Disease: Tomato_Bacterial_spot___Precaution: Use pathogen free seeds', 'Status: Tomato_healthy: No disease'];
#path1 = 'E:\\khk\\volumeF\\soft1\\WinPython\\settings\\.spyder-py3\\train\\nondigit'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
#onlyfiles1 = [f for f in listdir(path1) if isfile(join(path1, f))]

#img = cv2.imread('train/31762.0.png',1)
#img = cv2.imread('train/31893.5.png',1)
#img = cv2.imread('train/nondigit/'+random.choice(onlyfiles1),1)
#for i in onlyfiles:
for i in onlyfiles:
    t=path+'\\'+i
    full=cv2.resize(cv2.imread(t,1),(800,500))
    img1 = cv2.imread(t,1)
    img1=cv2.resize(img1, (224, 224))
    img1=np.array(img1)/255
    #simg1=cv2.resize(cv2.imread(path+"\\"+dire+"\\"+files,1), (80, 80))
    
    #print('train/digit/'+random.choice(onlyfiles))
    '''
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    dat=255-np.array([img[:,:,1:2],img[:    ,:,0:1]])
    avg=np.average(dat[0])
    print(np.average(dat[0]))
    dat=((dat<avg)*0)+((dat>avg)*dat)
    dit=dat[1][:,:,0]
    rddat=img[:,:,1]
    #image = cv2.cvtColor(np.array(pyautogui.screenshot(region=(0, 700, 500, 28))),cv2.COLOR_RGB2BGR)
    print(dat.shape)'''
    
    
    img=np.array([img1,img1])
    model = load_model('modelonetomato.h5')
    resu=model.predict_classes(img)
    print(resu)
    if(resu[0]>=0):
        cv2.putText(full,''+str(disease[resu[0]]),(0,20), 0, .51,(500,255,0),1,cv2.LINE_AA)
    else:
        cv2.putText(full,'error',(0,20), 0, .51,(500,255,0),1,cv2.LINE_AA)
    
    cv2.imshow('image',full)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(resu)
    
