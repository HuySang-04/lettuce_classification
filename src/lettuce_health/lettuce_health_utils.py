import math
import os
from keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras import models, layers
from keras.applications import VGG19


""" ======================================================================
                            create model
====================================================================== """
def create_model(image_size=224):
    base_model = VGG19(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
    # base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAvgPool2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

""" ======================================================================
                            lr_scheduler
====================================================================== """
def lr_scheduler(epoch, lr):
    return lr if epoch < 10 else lr * math.exp(-0.1)

""" ======================================================================
                            get_callbacks
====================================================================== """
def get_callbacks():
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_schedule = callbacks.LearningRateScheduler(lr_scheduler)
    return [early_stop, lr_schedule]

""" ======================================================================
                            plot_training
====================================================================== """
def plot_training(history):
    os.makedirs('../../outputs/figures/lettuce_health', exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../../outputs/figures/lettuce_health/training_curve.png')
    plt.show()

""" ======================================================================
                            Confusion-matrix
====================================================================== """
def evaluate_model(model, test_gen, class_names):
    y_true = test_gen.classes
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_labels = (y_pred > 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_true, y_pred_labels))
    print(classification_report(y_true, y_pred_labels, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('../../outputs/figures/lettuce_health/confusion_matrix.png')
    plt.show()