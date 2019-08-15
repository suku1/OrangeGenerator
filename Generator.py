import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import array_to_img

import os
import sys
import time
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    np.random.RandomState(seed)
    tf.set_random_seed(seed)

def get_time_txt():
    #時間をテキストデータで取得
    tm = time.localtime()
    return '_'.join(map(str, tm[:6]))

def gen_image(model_name, noise, combined=True):
    dirname = os.path.dirname(__file__)
    #モデル読み込み
    model = load_model(os.path.join(dirname, 'models', model_name))
    #モデルから画像を生成
    imgs = model.predict(noise)
    images = []
    num = noise.shape[0]
    for i in range(num):
        images.append(array_to_img(imgs[i]))
    save_folder = os.path.join(dirname, model_name[:-3] + '_' + get_time_txt())
    #保存フォルダ生成
    os.makedirs(save_folder, exist_ok=True)
    for i, img in enumerate(images):
        #配列を画像データに変換
        name = model_name[:-3] + '_' + str(i) + '.png'
        #画像を保存
        img.save(os.path.join(save_folder, name))
        print(name + ' was saved.')
    
    #すべての画像を1枚にまとめて保存
    if combined:
        from PIL import Image

        size = (10, (num // 10) + (num % 10 != 0))
        isize = images[0].size
        outlinesize = (16, 16)
        outlinecolor = (0, 0, 0)

        width = isize[0] * size[0] + outlinesize[0] * (size[0] + 1)
        height = isize[1] * size[1] + outlinesize[1] * (size[1] + 1)
        bg = Image.new('RGB', (width, height), outlinecolor)
        im_count = 0
        for h in range(size[1]):
            pos_h = h * (isize[1] + outlinesize[1]) + outlinesize[1]
            for w in range(size[0]):
                pos_w = w * (isize[0] + outlinesize[0]) + outlinesize[0]
                bg.paste(images[im_count], (pos_w, pos_h))
                im_count += 1

        name = model_name[:-3] + '_combined.png'
        bg.save(os.path.join(save_folder, name))

    np.savetxt(os.path.join(save_folder, 'noise'), noise, delimiter=',')

if __name__ == '__main__':
    #シード値を設定
    #set_seed(1000)
    model_name = 'Orange02.h5'
    num_image = 100
    l = len(sys.argv)
    if l >= 2: num_image = sys.argv[1]
    #画像num_image枚分のノイズを生成
    noise = np.random.uniform(-1, 1, (num_image, 100))
    #画像を生成
    gen_image(model_name, noise)
