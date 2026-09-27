import os
import sys
import tensorflow as tf

# Allow running as `python src/train.py` from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model, compile_model                       # noqa: E402
from preprocess import load_training_data, load_validation_data    # noqa: E402

TRAIN_DIR = os.environ.get('TRAIN_DIR', 'data/train')
VALID_DIR = os.environ.get('VALID_DIR', 'data/valid')
EPOCHS = int(os.environ.get('EPOCHS', 50))
MODEL_SAVE_PATH = os.environ.get(
    'MODEL_SAVE_PATH',
    os.path.join('saved_model', 'plant_disease_model.h5'),
)


def train():
    training_set = load_training_data(TRAIN_DIR)
    validation_set = load_validation_data(VALID_DIR)

    model = build_model(num_classes=38)
    model = compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    training_history = model.fit(
        x=training_set,
        validation_data=validation_set,
        epochs=EPOCHS,
        validation_steps=len(validation_set),
        callbacks=callbacks,
    )

    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    train_loss, train_acc = model.evaluate(training_set, verbose=1)
    val_loss, val_acc = model.evaluate(validation_set, verbose=1)
    print(f'Training accuracy: {train_acc:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}')

    return training_history


if __name__ == '__main__':
    train()