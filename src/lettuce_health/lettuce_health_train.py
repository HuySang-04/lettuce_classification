import argparse
from tensorflow.keras.optimizers import Adam
from lettuce_health_data_loader import load_data
from lettuce_health_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train lettuce_project health classifier')
    parser.add_argument('--data_path', type=str, default='../../data/plant-health')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)

    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    train_gen, val_gen, class_names = load_data(args.data_path, val_split=args.test_size, image_size=args.image_size,
                                                batch_size=args.batch_size, random_state=args.random_state)

    # Load model
    model = create_model(args.image_size)
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Trainning
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=get_callbacks()
    )

    # Visualization
    plot_training(history)
    evaluate_model(model, val_gen, class_names)

    model.save("../../outputs/lettuce_healthy_classification.keras")

if __name__ == "__main__":
    main()