import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

import time
import sys

# Define constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 20
LEARNING_RATE = 0.0001
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 2  # Healthy and not healthy
NUM_EPOCHS = 25

# Function to load and preprocess image
def load_and_preprocess_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    return img, label

# Load images and labels using tf.data.Dataset.list_files
train_dataset = tf.data.Dataset.list_files("./datasets/Augmented/Trainfolder/Class0/*.jpg")
train_dataset = train_dataset.map(lambda x: (x, 0))  # Label for healthy class is 0
train_dataset = train_dataset.concatenate(tf.data.Dataset.list_files("./datasets/Augmented/Trainfolder/Class1/*.jpg").map(lambda x: (x, 1)))  # Label for sick class is 1
train_dataset = train_dataset.map(load_and_preprocess_image)

val_dataset = tf.data.Dataset.list_files("./datasets/Augmented/Valfolder/Class0/*.jpg")
val_dataset = val_dataset.map(lambda x: (x, 0))  # Label for healthy class is 0
val_dataset = val_dataset.concatenate(tf.data.Dataset.list_files("./datasets/Augmented/Valfolder/Class1/*.jpg").map(lambda x: (x, 1)))  # Label for sick class is 1
val_dataset = val_dataset.map(load_and_preprocess_image)


# Define learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 150:
        return lr
    elif epoch < 225:
        return lr * 0.1
    else:
        return lr * 0.01

# Create learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create base model using pre-trained ResNet50 without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze base model layers
base_model.trainable = False

# Add new classification layers
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15), lr_scheduler]


start_time = time.time()
# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)
end_time = time.time()

total_time = end_time-start_time

print(f"Total time taken to train for {NUM_EPOCHS} epochs: {total_time}")
print(f"Time per epoch: {total_time/NUM_EPOCHS}")

timestr = time.strftime("%Y%m%d-%H%M%S")

# Save the model
model.save("resnet50_model"+timestr+".h5")


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Resnet Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Resnet Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Loss, Accuracy
loss, acc = model.evaluate(val_dataset)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {acc}")
