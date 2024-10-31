import os
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set directories and load labels
train_dir = './data/train'
labels_path = './data/labels.csv'

# Load labels and add '.jpg' to match image filenames
labels_df = pd.read_csv(labels_path)
labels_df['id'] = labels_df['id'].astype(str) + '.jpg'

# Initialize ImageDataGenerator for training and validation sets
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create data generators
train_gen = datagen.flow_from_dataframe(
    labels_df,
    directory=train_dir,
    x_col='id',
    y_col='breed',
    target_size=(128, 128),
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    labels_df,
    directory=train_dir,
    x_col='id',
    y_col='breed',
    target_size=(128, 128),
    class_mode='categorical',
    subset='validation'
)

# Save class names for later use
with open('class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(train_gen.class_indices), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[stop])

# Print the final training results
print(f"Training Accuracy: {history.history['accuracy'][-1]:.2f}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")

# Save the trained model
model.save('dog_breed_classifier_model.h5')
