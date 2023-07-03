from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from django.conf import settings


def home(request):
    if request.method == 'POST' and request.FILES['imagen']:
        imagen = request.FILES['imagen']
        modelo = load_model('main_page/CancerDetector_Final.h5')


        image_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_image.jpg')
        with open(image_path, 'wb') as f:
            for chunk in imagen.chunks():
                f.write(chunk)


        img = image.load_img(image_path, target_size=(150, 150), color_mode='grayscale')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        prediccion = modelo.predict(img)
        resultado = 'Cerebro sano (healthy)' if prediccion > 0.5 else 'Cerebro con tumor (tumor)'


        os.remove(image_path)

        return render(request, 'index.html', {'prediction': resultado})

    return render(request, 'index.html')
