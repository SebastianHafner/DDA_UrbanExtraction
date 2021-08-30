import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import losses
from tensorflow.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.keras.utils import plot_model

from tensorflow_implementation.data_generator import DataGenerator
from tensorflow_implementation.model import unet, UNet
from tensorflow_implementation.callbacks import *
from tensorflow_implementation.losses import *
from pathlib import Path
import matplotlib.pyplot as plt
import os
from os import path
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config





def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'../configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file

    if args.log_dir:  # Override Output dir
        cfg.OUTPUT_DIR = path.join(args.log_dir, args.config_file)
    else:
        cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg


# Segmentation tutorial https://www.tensorflow.org/tutorials/images/segmentation
if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)

    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)


    # Generators
    training_generator = DataGenerator(cfg, 'train')
    validation_generator = DataGenerator(cfg, 'test')

    # U-Net model
    model = UNet()
    model.build((None, 256, 256, 8))
    model.summary()

    model.compile(
        optimizer='adam',
        loss=bce_dice_loss,
        metrics=[dice_loss]
    )

    train_display = DisplayCallback(model, training_generator, [0, 1, 2])
    train_eval = NumericEvaluationCallback(model, training_generator, max_samples=100)
    callbacks = [train_display, train_eval]

    # Train model on dataset
    model.fit(
        x=training_generator,
        epochs=cfg.TRAINER.EPOCHS,
        use_multiprocessing=False,
        workers=cfg.DATALOADER.NUM_WORKER,
        # validation_data=validation_generator,
        verbose=True,
        callbacks=callbacks
    )

    model.evaluate(
        x=validation_generator,
        use_multiprocessing=False,
        workers=cfg.DATALOADER.NUM_WORKER,
        verbose=True
    )

    # saving network
    save_path = Path(cfg.OUTPUT_BASE_DIR) / cfg.NAME
    save_path.mkdir(exist_ok=True)
    net_file = save_path / 'tf_net'
    model.save_weights(str(net_file))
