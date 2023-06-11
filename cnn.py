from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Flatten,MaxPool2D,Dense,Dropout,GlobalAveragePooling2D

train_gen = ImageDataGenerator(rescale=1./255)
train = train_gen.flow_from_directory('dataset/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 color_mode='rgb',
                                                 class_mode = 'categorical')
test_gen = ImageDataGenerator(rescale=1./255)

test = test_gen.flow_from_directory('dataset/test',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 color_mode='rgb',
                                                 class_mode = 'categorical')

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding='same',activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=1))
model.add(Conv2D(15,kernel_size=(3,3),strides=1,padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=1))
model.add(Flatten())
model.add(Dense(140,activation="relu",kernel_initializer="he_normal"))
model.add(Dense(2,activation="sigmoid",kernel_initializer="uniform"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.summary()
model.fit(train,validation_data=test,epochs=10,batch_size=32)


print(model.evaluate(train))

model.save("face_mask.h5")
