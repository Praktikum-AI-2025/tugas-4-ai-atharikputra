# =====================================================================================================
# Membangun sebuah model Neural Network untuk klasifikasi dataset Horse or Human dalam binary classes.
#
# Input layer harus menerima 150x150 dengan 3 bytes warna sebagai input shapenya.
# Jangan menggunakan lambda layers dalam model.
#
# Dataset yang digunakan dibuat oleh Laurence Moroney (laurencemoroney.com).
#
# Standar yang harus dicapai untuk accuracy dan validation_accuracy > 83%
# =====================================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def solution_05():
    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human'
    VALIDATION_DIR = 'data/validation-horse-or-human'

    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_generator= ImageDataGenerator(rescale=1./255)
    
    # YOUR CODE HERE
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    VALIDATION_DIR='data/validation-horse-or-human'

     # Hanya rescale untuk validation set 
    validation_datagen=ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    model=tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
        Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')  # DO NOT CHANGE THIS LINE!
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',  # Monitoring validation accuracy
        patience=10,
        restore_best_weights=True,  # This ensures we keep the best weights
        verbose=1  # To see when early stopping occurs
    )

    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=[early_stop],
        verbose=1
    )

    return model
    # Get training metrics from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    final_train_acc = acc[-1]
    final_val_acc = val_acc[-1]

    # NEW: Get the best validation accuracy during training (what EarlyStopping saved)
    best_val_acc = max(val_acc)
    best_acc = max(acc)
    best_epoch = val_acc.index(best_val_acc) + 1  # +1 because epochs start at 1

    print('\nTraining Results:')
    print(f'Final Training Accuracy: {final_train_acc*100:.2f}%')
    print(f'Final Validation Accuracy: {final_val_acc*100:.2f}%')
    print(f'\nBest Accuracy: {best_acc*100:.2f}% (epoch {best_epoch})')
    print(f'Best Validation Accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})')

    if best_val_acc > 0.83:
        print("\nModel mencapai target validation accuracy > 83%!")
    else:
        print("\nModel belum mencapai target. Rekomendasi:")
        print("- Tingkatkan jumlah epoch (misal: 20-25)")

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_05()
    model.save("model_05.h5")
