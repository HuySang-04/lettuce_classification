import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime
from lettuce_disease_utils import CLASS_NAMES

def parser_args():
    parser = argparse.ArgumentParser(description='Test lettuce_project disease classifier')
    parser.add_argument('--test_path', type=str, default='../../data/lettuce_disease_test')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model_path', type=str, default='../../outputs/lettuce_disease_vgg19_classification.keras')

    return parser.parse_args()

def test():
    args = parser_args()

    model = load_model(args.model_path)
    os.makedirs('../../outputs/lettuce_disease_test', exist_ok=True)

    if os.path.isfile(args.test_path):
        image = Image.open(args.test_path).convert('RGB')
        ori_image = image.copy()
        image = image.resize((args.image_size, args.image_size))
        image = np.array(image) /255.0
        image = np.expand_dims(image, axis=0)

        predict = model.predict(image)
        pred_id = np.argmax(predict, axis=1)

        plt.imshow(ori_image)
        plt.title(CLASS_NAMES[pred_id[0]])
        plt.savefig(f"../../outputs/lettuce_disease_test/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

    else:

        for img_file in os.listdir(args.test_path):
            image = Image.open(os.path.join(args.test_path,img_file)).convert('RGB')
            ori_image = image.copy()
            image = image.resize((args.image_size, args.image_size))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            predict = model.predict(image)
            pred_id = np.argmax(predict, axis=1)

            plt.imshow(ori_image)
            plt.title(f'{img_file} -- {CLASS_NAMES[pred_id[0]]}')
            plt.savefig(f"../../outputs/lettuce_disease_test/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

if __name__ == '__main__':
    test()
