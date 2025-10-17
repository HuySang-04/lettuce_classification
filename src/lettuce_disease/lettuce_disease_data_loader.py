import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from lettuce_disease_utils import CLASS_NAMES

def load_data(data_path='../../data/Lettuce_disease_datasets', val_split=0.2, image_size=224, batch_size=32, random_state=42):
    image_paths, labels = [], []

    for class_name in os.listdir(data_path):
        for img in os.listdir(f'{data_path}/{class_name}'):
            image_paths.append(f'{data_path}/{class_name}/{img}')
            labels.append(class_name)

    df = pd.DataFrame({"image_path": image_paths, "label": labels})

    df_train, df_val = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=random_state)

    train_set = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        df_train,
        x_col='image_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        classes=CLASS_NAMES,
    )

    val_set = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        df_val,
        x_col='image_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        classes=CLASS_NAMES,
    )
    return train_set, val_set