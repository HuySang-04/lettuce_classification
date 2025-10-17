import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from datetime import datetime
import numpy as np

data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

def augmentation_image(name_dir, image, num_aug=3):
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.stack([image_array]*3, axis=-1)
    elif image_array.shape[2] != 3:
        return
    image_array = np.expand_dims(image_array, axis=0)
    aug_iter = data_gen.flow(image_array, batch_size=1)

    for _ in range(num_aug):
        aug_img_array = next(aug_iter)[0].astype(np.uint8)
        aug_img = Image.fromarray(aug_img_array)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{np.random.randint(0,10000)}.jpg"
        save_path = os.path.join(name_dir, filename)
        try:
            aug_img.save(save_path)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Cannot save {save_path}: {e}")

def create_more_data(data_path='../../data/Lettuce_disease_datasets'):
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    for class_name in class_dirs:
        name_dir = os.path.join(data_path, class_name)
        print(f"\nProcessing class: {class_name}")
        img_files = [f for f in os.listdir(name_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in img_files:
            img_path = os.path.join(name_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Cannot open {img_path}: {e}")
                continue

            if 'Wilt_and_leaf_blight_on_lettuce' in class_name:
                num_aug = 5
            else:
                num_aug = 3

            augmentation_image(name_dir, image, num_aug=num_aug)

    print("\nAugmentation completed.")
