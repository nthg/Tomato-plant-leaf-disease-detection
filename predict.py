import pandas as pd
import cv2
import numpy as np 
import random
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
#%matplotlib inline
import math 
import datetime
import time
img_width, img_height = 224, 224
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'C:/Users/Hp/Desktop/Train'
validation_data_dir = 'C:/Users/Hp/Desktop/Validation'
test_data_dir = 'C:/Users/Hp/Desktop/test'
epochs = 7
batch_size = 50
vgg16 = applications.VGG16(include_top=False,weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)
start = datetime.datetime.now()
 
generator = datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_train_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
 
predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
 
bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train) 
 
np.save('bottleneck_features_train.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)
#training data
generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
train_data = np.load('bottleneck_features_train.npy') 
 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)

#Validation data
generator_top = datagen.flow_from_directory( 
   validation_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_validation_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
validation_data = np.load('bottleneck_features_train.npy') 
 
# get the class labels for the training data, in the original order 
validation_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

#testing data
generator_top = datagen.flow_from_directory( 
   test_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_test_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
test_data = np.load('bottleneck_features_train.npy') 
 
# get the class labels for the training data, in the original order 
test_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
test_labels = to_categorical(test_labels, num_classes=num_classes)
#This is the best model we found. For additional models, check out I_notebook.ipynb
start = datetime.datetime.now()
model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
history = model.fit(train_data, train_labels, 
   epochs=7,
   batch_size=batch_size, 
   validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)
(eval_loss, eval_accuracy) = model.evaluate( 
    validation_data, validation_labels, batch_size=batch_size,verbose=1)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
print("[INFO] Loss: {}".format(eval_loss)) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

#plotting graphs
#Graphing our training and validation
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('ACCURACY : Training and Validation')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('LOSS : Training and Validation')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()

model.evaluate(test_data, test_labels)
'''
#CLASSIFICATION METRICS
preds = np.round(model.predict(test_data),0)
#to fit them into classification metrics and confusion metrics, some additional modifications are required
print('rounded test_labels', preds)

animals = ['butterflies', 'chickens', 'elephants', 'horses', 'spiders', 'squirells']
classification_metrics = metrics.classification_report(test_labels, preds, target_names = animals)
print(classification_metrics)

#Since our data is in dummy format we put the numpy array into a dataframe and call idxmax axis=1 to return the column
# label of the maximum value thus creating a categorical variable
#Basically, flipping a dummy variable back to it’s categorical variable
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues):
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')
 
# print(cm)
 
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
       plt.tight_layout()
       plt.ylabel('True label')
       plt.xlabel('Predicted label') 
       plot_confusion_matrix(confusion_matrix,['butterflies', 'chickens', 'elephants', 'horses', 'spiders', 'squirells'],normalize=True)
    '''
   #FINAL PHASE IN TESTING
def read_image(file_path):
    print("[INFO] loading and preprocessing image…") 
    image = load_img(file_path, target_size=(224, 224)) 
    image = img_to_array(image) 
    image = np.expand_dims(image, axis=0)
    image /= 255. 
    return image
def test_single_image(path):
  animals = ['butterflies', 'chickens', 'elephants', 'horses', 'spiders', 'squirells']
  images = read_image(path)
  time.sleep(.5)
  bt_prediction = vgg16.predict(images) 
  preds = model.predict_proba(bt_prediction)
  for idx, animal, x in zip(range(0,6), animals , preds[0]):
      
      print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
      print('Final Decision:')
      time.sleep(.5)
  for x in range(3):
   print('.'*(x+1))
   time.sleep(.2)
  class_predicted = model.predict_classes(bt_prediction)
  class_dictionary = generator_top.class_indices 
  inv_map = {v: k for k, v in class_dictionary.items()} 
  print("ID: {}, Label: {}".format(class_predicted[0],  inv_map[class_predicted[0]])) 
  return load_img(path)
path = 'data/test/yourpicturename'
test_single_image(path)
#https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324f
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
