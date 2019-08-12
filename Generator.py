import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import array_to_img

import os
import time
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    np.random.RandomState(seed)
    tf.set_random_seed(seed)

def get_time_txt():
    tm = time.localtime()
    return '_'.join(map(str, tm[:6]))

def gen_image(model_name, noise):
    dirname = os.path.dirname(__file__)
    model = load_model(os.path.join(dirname, 'models', model_name))
    images = model.predict(noise)
    save_folder = os.path.join(dirname, model_name + '_' + get_time_txt())
    os.makedirs(save_folder, exist_ok=True)
    for i, img_ary in enumerate(images):
        img = array_to_img(img_ary)
        name = model_name[:-3] + '_' + str(i) + '.png'
        img.save(os.path.join(save_folder, name))
        print(name + ' was saved.')

if __name__ == '__main__':
    #set_seed(1000)
    model_name = 'Orange02.h5'
    num_image = 100
    noise = np.random.uniform(-1, 1, (num_image, 100))
    gen_image(model_name, noise)
