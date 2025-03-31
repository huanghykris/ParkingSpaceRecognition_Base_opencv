import tensorflow as tf
import numpy
import os
from keras import applications, Model
from keras import optimizers
from keras import backend as k
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator

files_train = 0
files_validation = 0

cwd = os.getcwd()
folder = 'train_data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_train += len(files)


folder = 'train_data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_validation += len(files)

print(files_train,files_validation)

img_width, img_height = 48, 48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2

model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))


for layer in model.layers[:10]:
    layer.trainable = False


x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)


model_final = Model(inputs = model.input, outputs = predictions)


model_final.compile(optimizer = optimizers.Adam(learning_rate=0.0001, ema_momentum=0.9),
                    loss = "categorical_crossentropy",
                    metrics=["accuracy"]) 


train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

checkpoint = ModelCheckpoint(filepath="car1.model.keras", monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='max')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')




history_object = model_final.fit(
train_generator,
steps_per_epoch=nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
validation_steps=nb_validation_samples,
callbacks = [checkpoint, early])