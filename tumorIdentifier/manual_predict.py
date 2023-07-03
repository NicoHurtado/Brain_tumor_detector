import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import glob
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

modelo = tf.keras.models.load_model('main_page/CancerDetector_Final.h5')
print(modelo.summary())
capa_salida = modelo.output
print(capa_salida)

def predict(ruta_imagen):

    ruta_imagen = ruta_imagen

    img = image.load_img(ruta_imagen, target_size=(150, 150), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediccion = modelo.predict(img)
    val_prediccion = prediccion[0][0]
    print(val_prediccion)

    if prediccion  > 0.5:
        print("La imagen corresponde a un cerebro -SANO-")
    else:
        print("La imagen corresponde a un cerebro -CON TUMOR-")

def create_samples():
    directorio_h = 'Sample/sample_healthy'
    directorio_c = 'Sample/sample_cancer'

    carpeta_origenh = 'Brain Tumor Data Set/Healthy'
    carpeta_origenc = 'Brain Tumor Data Set/Brain Tumor'

    patron_archivosh = carpeta_origenh + '/*.jpg'
    patron_archivosc = carpeta_origenc + '/*.jpg'

    lista_archivosH = sorted(glob.glob(patron_archivosh))[:2000]
    lista_archivosc = sorted(glob.glob(patron_archivosc))[:2000]

    for archivo in lista_archivosH:
        shutil.copy(archivo, directorio_h)

    for archivo in lista_archivosc:
        shutil.copy(archivo, directorio_c)

def evaluate_model():

    nuevo_directorio = 'Sample'
    sample = ImageDataGenerator(rescale=1./255)

    imagenes_prueba = sample.flow_from_directory(
        nuevo_directorio,
        target_size=(150, 150),
        batch_size=20,
        color_mode='grayscale',
        class_mode='binary',
        shuffle=False
    )

    resultado = modelo.evaluate(imagenes_prueba)
    print("Pérdida:", resultado[0])
    print("Precisión:", resultado[1])




