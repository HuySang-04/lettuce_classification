from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19, InceptionV3,ResNet50
from keras import models, layers
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


"""=================================================
                    load_data.py
================================================="""
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    brightness_range=[0.8, 1.3],
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

"""=================================================
                    train.py
================================================="""
def create_mode(model_name, image_size):
    input_shape = (image_size, image_size, 3)
    model = None

    if model_name =='VGG19':
        # ==================== VGG19 ====================
        base_model = VGG19(include_top=False, input_shape=input_shape, weights="imagenet")
        base_model.trainable=False
        for layer in base_model.layers[-10:]:
            layer.trainable = True

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-5)),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    elif model_name == 'InceptionV3':
        base_model = InceptionV3(include_top=False, input_shape=input_shape, weights="imagenet")
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
    elif model_name == 'ResNet50':
        base_model = ResNet50(include_top=False, input_shape=input_shape, weights="imagenet")
        base_model.trainable = False
        for layer in base_model.layers[-100:]:
            layer.trainable = True

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

    return model

# ==================== Visualization training ====================
def visualization_training(history, model_name, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title(f'{model_name} - Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title(f'{model_name} - Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    plt.savefig(save_path)

# ==================== Testing ====================
def testing(test_set, model, class_names, model_name, save_path):
    y_true = test_set.classes
    y_predict = model.predict(test_set, verbose=1)
    y_predict_labels = (y_predict > 0.5).astype(int).flatten()

    acc = accuracy_score(y_true, y_predict_labels)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_predict_labels, target_names=class_names))

    cm = confusion_matrix(y_true, y_predict_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                linewidths=0.3, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(save_path)