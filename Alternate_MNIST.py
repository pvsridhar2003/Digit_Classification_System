# Alternate Program with higher Accuracy but Slower Model Training


import tensorflow as tf
import plotly.express as px
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

data = random.randint(0,60000)
fig = px.imshow(x_train[data])
fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Sample MNIST Dataset') 
fig.show()

x_train = x_train.astype("float32")/255 
x_test = x_test.astype("float32")/255
x_train = tf.convert_to_tensor(x_train) 
x_test = tf.convert_to_tensor(x_test)
# Add an extra dimension for the channel 
x_train = tf.expand_dims(x_train, axis=-1) 
x_test = tf.expand_dims(x_test, axis=-1)

# Check the shapes
print("x_train shape:", x_train.shape) 
print("x_test shape:", x_test.shape)

#Conv2d data_format parameter we use 'channel_last' for imgs
model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',  strides=1,  padding='same',  data_format='channels_last', input_shape=(28,28,1)))
model.add(tf.keras.layers.BatchNormalization())  
model.add(tf.keras.layers.Conv2D(filters=32,  kernel_size=(3,  3), activation='relu',  strides=1,  padding='same',  data_format='channels_last'))
model.add(tf.keras.layers.BatchNormalization())  
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last')) 
model.add(tf.keras.layers.BatchNormalization())  
model.add(tf.keras.layers.Conv2D(filters=64,  kernel_size=(3,  3),  strides=1, padding='same',  activation='relu',  data_format='channels_last'))
model.add(tf.keras.layers.BatchNormalization())  
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(512, activation='relu')) 
model.add(tf.keras.layers.BatchNormalization())  
model.add(tf.keras.layers.Dropout(0.25)) 
model.add(tf.keras.layers.Dense(1024,  activation='relu')) 
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])

num_classes=10
#  Convert  class  vectors  to  binary  class  matrices  (one-hot  encoding) 
y_train  =  tf.keras.utils.to_categorical(y_train,  num_classes) 
y_test = tf.keras.utils.to_categorical(y_test, num_classes) 
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#visualizing what the learning rate decay will do to the learning rate through every epoch
decays = [(lambda x: 1e-3 * 0.9 ** x)(x) for x in range(10)] 
i=1
#for our case LearningRateScheduler will work great
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
for lr in decays:
  print("Epoch " + str(i) +" Learning Rate: " + str(lr)) 
  i+=1
  early_stopping  =  tf.keras.callbacks.EarlyStopping(
      min_delta=0.001, # minimium amount of change to count as an improvement 
      patience=20, # how many epochs to wait before stopping 
      restore_best_weights=True
      )
#defining these prior to model to increase readability and debugging
batch_size = 64
epochs = 50
history = model.fit(x=x_train, y=y_train, epochs = epochs, validation_data  =  (x_test,  y_test), verbose=1, callbacks = [reduce_lr])

model.save('./mnist_detection_model.h5') 

model = tf.keras.models.load_model('./mnist_detection_model.h5') 
print('Model loaded Sucessfully')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

test_data = random.randint(0,10000)
image = x_test[test_data] 
fig = px.imshow(image)
fig.update_layout(width=600, height=500,	margin=dict(l=10, r=10, b=10, t=10), xaxis_title='TEST Image') 
fig.show()
print(image.shape)

image = tf.expand_dims(image, axis=-1) 
image = image.numpy()
image = image.reshape(1,28,28,1)

coords = model.predict(image)

if 1 in coords[0]:
  array = list(coords[0])
  print("Predicted value: ", array.index(1))
else:
  print('Cannot predict')
