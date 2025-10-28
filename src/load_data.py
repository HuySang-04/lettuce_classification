from sklearn.model_selection import train_test_split
from utils import train_datagen, test_val_datagen
import pandas as pd
import os


def load_dataset(data_path, batch_size, class_names, image_size, test_size, val_size, random_state):
    img_paths = []
    labels = []

    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            img_paths.append(os.path.join(label_path, file))
            labels.append(label)


    df = pd.DataFrame({"img_path": img_paths, "label": labels})

    print('\n=================== Load dataset ===================')
    print('Number of dataset: ', len(df))

    x_train_val, x_test = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )

    x_train, x_val = train_test_split(
        x_train_val, test_size=val_size, stratify=x_train_val['label'], random_state=random_state
    )

    print('Number of train: ', len(x_train))
    print('Number of val: ', len(x_val))
    print('Number of test: ', len(x_test))

    train_set = train_datagen.flow_from_dataframe(
        x_train,
        x_col='img_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=class_names,
        shuffle=True,
    )

    val_set = test_val_datagen.flow_from_dataframe(
        x_val,
        x_col='img_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=class_names,
    )

    test_set = test_val_datagen.flow_from_dataframe(
        x_test,
        x_col='img_path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=class_names,
    )

    return train_set, val_set, test_set