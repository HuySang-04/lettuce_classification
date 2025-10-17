import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime

def parser_args():
    parser = argparse.ArgumentParser(description='Test lettuce_project healthy classifier')
    parser.add_argument('--test_path', default= '/home/sang/Practice/python/lettuce_project/data/plant-health/healthy/20230622104701.jpg', type=str)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model_path', type=str, default='../../outputs/lettuce_health_VGG19_classification.keras')

    return parser.parse_args()

def test():
    args = parser_args()

    model = load_model(args.model_path)
    class_name =['unhealthy', 'healthy']
    os.makedirs('../../outputs/lettuce_health_test', exist_ok=True)

    if os.path.isfile(args.test_path):
        image = Image.open(args.test_path).convert('RGB')
        ori_image = image.copy()
        image = image.resize((args.image_size, args.image_size))
        image = np.array(image) /255.0
        image = np.expand_dims(image, axis=0)

        predict = model.predict(image)
        predict_labels = (predict[0] > 0.5).astype(int)

        plt.imshow(ori_image)
        plt.title(class_name[predict_labels[0]])
        plt.savefig(f"../../outputs/lettuce_health_test/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

    else:
        for img_file in os.listdir(args.test_path):
            image = Image.open(os.path.join(args.test_path,img_file)).convert('RGB')
            ori_image = image.copy()
            image = image.resize((args.image_size, args.image_size))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            predict = model.predict(image)
            predict_labels = (predict[0] > 0.5).astype(int)

            plt.imshow(ori_image)
            plt.title(class_name[predict_labels[0]])
            plt.savefig(f"../../outputs/lettuce_health_test/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

if __name__ == '__main__':
    test()
