import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_path='../../data/plant-health', val_split=0.2, image_size=224, batch_size=32, random_state=42):
    """Tao dataframe va ImageDataGenerator cho train/val"""
    image_paths, labels = [], []
    class_names = ['unhealthy', 'healthy']

    for class_name in os.listdir(data_path):
        class_dir = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_name)

    df = pd.DataFrame({'image_path': image_paths, 'label': labels})

    # Chia du lieu train/val
    df_train, df_val = train_test_split(df, test_size=val_split,
                                        stratify=df['label'], random_state=random_state)

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col='image_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=class_names,
    )

    val_gen = val_datagen.flow_from_dataframe(
        df_val,
        x_col='image_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=class_names,
    )

    return train_gen, val_gen, class_names