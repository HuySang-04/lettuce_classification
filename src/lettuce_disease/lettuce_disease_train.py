import os
import pandas as pd
import argparse
from keras.optimizers import Adam
from lettuce_disease_utils import *
from lettuce_disease_create_dataset import create_more_data
from lettuce_disease_data_loader import load_data
from keras.losses import CategoricalCrossentropy

def parse_args():
    parser = argparse.ArgumentParser(description='Train lettuce_project health classifier')
    parser.add_argument('--data_path', type=str, default='../../data/Lettuce_disease_datasets')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--create_data', type=bool, default=False, help='The first time you run should value = True, else False')
    return parser.parse_args()

def main():
    args = parse_args()

    # create more data
    if args.create_data:
        create_more_data(data_path=args.data_path)

    # Load data
    train_set, val_set = load_data(args.data_path, val_split=args.test_size, image_size=args.image_size,
                                                batch_size=args.batch_size, random_state=args.random_state)

    # Load model
    model = create_model(args.image_size)
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Trainning
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=args.epochs,
        callbacks=get_callbacks()
    )

    # Visualization
    plot_training(history)
    evaluate_model(model, val_set, CLASS_NAMES)

    model.save("../../outputs/lettuce_disease_classification.keras")

if __name__ == "__main__":
    main()