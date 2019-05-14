
# coding: utf-8

# In[3]:


import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import keras
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# In[4]:


base_model=MobileNet(weights='imagenet',include_top=False) 
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.35)(x)
x=Dense(256,activation='relu')(x) 
x=Dropout(0.25)(x)
x=Dense(128,activation='relu')(x) 
preds=Dense(1,activation='sigmoid')(x) 


# In[5]:


model=Model(inputs=base_model.input,outputs=preds)
for layer in model.layers[:17]:
    layer.trainable=False
for layer in model.layers[17:]:
    layer.trainable=True


# In[11]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

train_generator=train_datagen.flow_from_directory(DATADIR, 
                                                 target_size=(100,100),
                                                 color_mode='rgb',
                                                 batch_size=128,
                                                 class_mode='binary',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size


# In[12]:


model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=3)


# In[ ]:



model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

