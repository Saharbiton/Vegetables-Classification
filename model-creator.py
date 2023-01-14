import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.__version__
train_path = ""

#train set
train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
                    )

training_set = train_datagen.flow_from_directory(
                    "D:/vegetable-classifier/Vegetable Images/train",
                    target_size = (64, 64),
                    batch_size = 32,
                    class_mode = "categorical"
                    )

#test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
                    "D:/vegetable-classifier/Vegetable Images/test",
                    target_size = (64, 64),
                    batch_size = 32,
                    class_mode = "categorical"
                    )

#validate set
validate_datagen = ImageDataGenerator(rescale = 1./255)
validate_set = validate_datagen.flow_from_directory(
                    "Vegetable Images/validation",
                    target_size = (64, 64),
                    batch_size = 32,
                    class_mode = "categorical"
                    )


#intitializing
cnn = tf.keras.models.Sequential()
#convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size =  3, activation= "relu", input_shape = (64, 64, 3)))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#second convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size =  3, activation= "relu"))
#second pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 512, activation="relu"))
cnn.add(tf.keras.layers.Dense(units = 128, activation="relu"))
cnn.add(tf.keras.layers.Dense(units = 64, activation="relu"))
cnn.add(tf.keras.layers.Dense(units = 64, activation="relu"))

cnn.add(tf.keras.layers.Dense(units = 15, activation="softmax"))
#softmax because we are doing categorical classification

#compiling the cnn
cnn.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#training the cnn on train set
# cnn.fit(x = training_set, validation_data = test_set, epochs = 16 )
cnn.fit(x = training_set, validation_data = validate_set, epochs = 16 )

cnn.save("models/vegitable_predictor")

