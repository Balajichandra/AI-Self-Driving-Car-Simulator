from utils import *
from sklearn.model_selection import train_test_split
import time
from keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Step 1: Setting path 
path ="C:/Users/Balaji/Documents/Machine Learning/Deep Learning/Self Driving cars/New folder (2)/New Data"
#data = ImportDataInfo(path)
data = importDataInfo(path)
# Step 2: Visualization and Balance data
#data = BalanceData(data,display=False)
data = balanceData(data,display=True)
#Step 3: Preparing for processing
imagePath,steering = loadData(path,data)
#print(imagePath[0],steering[0])

#Step 4: spliting  data for trainig and validation
xtrain,xval,ytrain,yval = train_test_split(imagePath,steering,test_size=0.2,random_state=5)
print("Total training images:",len(xtrain))
time.sleep(1)
print("Total validation images:",len(xval))

model = createModel()
model.summary()

model.fit(batchGen(xtrain,ytrain,32,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xval,yval,32,0),validation_steps=200)

model.save('car_model.h5') 
         
         