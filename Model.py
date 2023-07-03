import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

directorio_principal = 'Brain Tumor Data Set' # this was the dataSet that i used to train the model (Is not in files bc it was so large)

tamano_imagen = (150, 150)

datagen = ImageDataGenerator(rescale=1./255)

conjunto_entrenamiento = datagen.flow_from_directory(
    directorio_principal,
    target_size=tamano_imagen,
    batch_size=32,
    color_mode="grayscale",
    class_mode='binary',
    subset='training'
)

modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(tamano_imagen[0], tamano_imagen[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = modelo.fit(
    conjunto_entrenamiento,
    epochs=10,
)

accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.show()


modelo.save('CancerDetector_Final.h5')
