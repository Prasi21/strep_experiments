import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Define constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 20
LEARNING_RATE = 0.0001
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 2  # Healthy and not healthy
NUM_EPOCHS = 250

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

# test_dataset = tf.data.Dataset.list_files("./datasets/Testfolder/Class0/*.jpeg")
# test_dataset = test_dataset.map(lambda x: (x, 0))  # Label for healthy class is 0
# test_dataset = test_dataset.concatenate(tf.data.Dataset.list_files("/content/Testfolder/Class1/*.jpeg").map(lambda x: (x, 1)))  # Label for sick class is 1
# test_dataset = test_dataset.map(load_and_preprocess_image)


# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import time
timestp = time.strftime("%Y%m%d-%H%M%S")
def finetune_pretrained(base_model, out_dir=f"model-{timestp}.keras"):
    # Freeze base model layers
    base_model.trainable = False


    # Add new classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001),)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=output)

    # Optionally, unfreeze some layers of the convolutional base for fine-tuning
    for layer in model.layers[:15]:
        layer.trainable = True


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [
        reduce_lr,
        early_stopping,]


    # Train the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

    # Save the model
    model.save(out_dir)
    return history, model



from tensorflow.keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
history_vgg, vgg_model = finetune_pretrained(base_model, out_dir=f"vgg16-{timestp}.pt")



