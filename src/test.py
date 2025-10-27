import argparse
import os
from  PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Train lettuce_project health classifier')
    parser.add_argument('--test_path', type=str, default='../data/lettuce_tests')
    parser.add_argument('--out_path', type=str, default='../outputs/test_out')
    parser.add_argument('--model_path', type=str,default='../outputs/lettuce_health_classify_InceptionV3.keras')
    parser.add_argument('--image_size', type=int, default=224)

    return parser.parse_args()

def test():
    args = get_args()

    class_names = ['healthy', 'unhealthy']
    model = load_model(args.model_path)
    os.makedirs(args.out_path, exist_ok=True)

    for file in os.listdir(args.test_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_path = os.path.join(args.test_path, file)

        img = Image.open(file_path).convert('RGB')
        ori_img = img.copy()
        img = img.resize((args.image_size, args.image_size))
        img_np = np.array(img)/255
        img_np = np.expand_dims(img_np, axis=0)

        y_predict = model.predict(img_np, verbose=1)
        y_predict_idx = (y_predict > 0.5).astype(int).flatten()

        plt.imshow(ori_img)
        plt.title(class_names[y_predict_idx[0]])
        plt.axis('off')
        plt.savefig(f'{args.out_path}/{file}')

if __name__ == "__main__":
    test()