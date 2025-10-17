import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from PIL import Image
from datetime import datetime

def test_image(image_path, image_size, model1, model2, save_dir='./outputs/test_output'):
    class_names_model1 = ['unhealthy', 'healthy']
    class_names_model2 = [
        'Bacterial', 'Downy_mildew_on_lettuce', 'Viral',
        'Powdery_mildew_on_lettuce', 'Wilt_and_leaf_blight_on_lettuce',
        'Septoria_blight_on_lettuce'
    ]

    class_names_model2_vi = ['Bệnh do vi khuẩn', 'Bệnh mốc xám trên xà lách',
                            'Bệnh do virus', 'Bệnh mốc bột trên xà lách',
                            'Bệnh héo và thán thư lá trên xà lách', 'Bệnh Septoria trên xà lách']

    os.makedirs(save_dir, exist_ok=True)

    image = Image.open(image_path).convert('RGB')
    ori_image = image.copy()
    image = image.resize((image_size, image_size))
    image = np.array(image)/255
    image = np.expand_dims(image, axis=0)

    predict1 = model1.predict(image)
    predict_labels = (predict1[0] > 0.5).astype(int)
    label1 = class_names_model1[predict_labels[0]]

    plt.imshow(ori_image)
    plt.title(f'{label1}')

    if label1 == 'unhealthy':
        predict2 = model2.predict(image)
        predict2 = np.array(predict2)[0]
        sorted_indices = predict2.argsort()[::-1]

        # if predict2[sorted_indices[0]] > 0.7:
        #     label2 = f'{class_names_model2[sorted_indices[0]]} ({predict2[sorted_indices[0]]:.2f})'
        # else:
        #     label2 = (f'{class_names_model2[sorted_indices[0]]} ({predict2[sorted_indices[0]]:.2f}),'
        #               f'{class_names_model2[sorted_indices[1]]} ({predict2[sorted_indices[1]]:.2f})')

        if predict2[sorted_indices[0]] > 0.7:
            label2 = f'{class_names_model2_vi[sorted_indices[0]]} ({predict2[sorted_indices[0]]:.2f})'
        else:
            label2 = (f'{class_names_model2_vi[sorted_indices[0]]} ({predict2[sorted_indices[0]]:.2f}),'
                      f'{class_names_model2_vi[sorted_indices[1]]} ({predict2[sorted_indices[1]]:.2f})')

        plt.xlabel(label2)

    plt.savefig(f"{save_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test lettuce_project image with 2-step model")
    parser.add_argument("--image", type=str, default='/home/sang/Practice/python/lettuce_project/data/lettuce_disease_test/11_Downy_mildew.jpg', help="Path to input image")
    parser.add_argument("--model1", type=str, default='../outputs/lettuce_health_VGG19_classification.keras', help="Path to model1 (healthy/unhealthy)")
    parser.add_argument("--model2", type=str,  default='../outputs/lettuce_disease_vgg19_classification.keras', help = "Path to model2 (disease classification)")
    parser.add_argument("--save_dir", type=str, default='../outputs/lettuce_health_disease_test', help="Directory to save output image")
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    model1 = load_model(args.model1)
    model2 = load_model(args.model2)

    test_image(args.image, args.image_size, model1, model2, save_dir=args.save_dir)