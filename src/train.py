import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
from load_data import load_dataset
from utils import create_mode, visualization_training, testing
from keras.optimizers import Adam
from keras import callbacks

def get_args():
    parser = argparse.ArgumentParser(description='Train lettuce_project health classifier')
    parser.add_argument('--data_path', type=str, default='../data/lettuce_health_dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--model', type=int, default=None, help='1: InceptionV3, 2: VGG19, 3: ResNet50')
    parser.add_argument('--patience', type=int, default=4, help='Patience threshold to stop training.')

    return parser.parse_args()

def train():
    args = get_args()
    class_names = ['healthy', 'unhealthy']
    learning_rates = [1e-5, 5e-6, 1e-5]

    train_set, val_set, test_set = load_dataset(
        data_path=args.data_path, batch_size=args.batch_size, image_size=args.image_size,
        class_names=class_names, test_size=args.test_size, val_size=args.val_size,random_state=args.random_state
    )

    if args.model == None or args.model == 1:
        lr = learning_rates[0]
        model_name = "InceptionV3"
    elif args.model == 2:
        lr = learning_rates[1]
        model_name = 'VGG19'
    else:
        lr = learning_rates[2]
        model_name = 'ResNet50'

    if args.learning_rate is not None:
        lr = args.learning_rate

    model = create_mode(model_name=model_name, image_size=args.image_size)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy']),

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=0
    )

    print('\n\n==================== Training ====================')

    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=args.epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    print('\n\n==================== Testing ====================')
    save_path='../outputs'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/figures', exist_ok=True)

    cm_save_path = os.path.join(save_path, f'figures/cm_{model_name}.jpg')
    testing(test_set=test_set, model=model, model_name=model_name,
            class_names=class_names, save_path=cm_save_path)

    model_save_path = os.path.join(save_path, f'lettuce_health_classify_{model_name}.keras')
    model.save(model_save_path)
    print(f'\nSuccess save model: ', model_save_path)

    save_visual_training_path = os.path.join(save_path, f'figures/visual_training_{model_name}.jpg')
    visualization_training(history, model_name, save_visual_training_path)
    print(f'Success save visualization training:', save_visual_training_path)

if __name__ == '__main__':
    train()