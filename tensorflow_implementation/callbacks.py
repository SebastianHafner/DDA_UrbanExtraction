import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_implementation.data_generator import DataGenerator
from tensorflow_implementation import metrics


class DisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, data_generator: DataGenerator, sample_indices: list = []):
        super(DisplayCallback).__init__()
        self.model = model
        self.data_generator = data_generator
        self.sample_indices = sample_indices
        self.n = len(sample_indices)

    def on_epoch_end(self, epoch, logs=None):
        self.display_samples()

    def on_batch_end(self, batch, logs=None):
        # self.display_samples()
        pass

    def display_samples(self):

        # Setting up fiture
        fig_height = self.n * 10
        fig_width = 30
        cols = 3
        fig, axs = plt.subplots(self.n, cols, figsize=(fig_width, fig_height))
        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        for i, sample_index in enumerate(self.sample_indices):
            img, label = self.data_generator.get_sample(sample_index)

            # Get model prediction
            building_prob = self.model.predict(img[None, ])
            building_prob = building_prob[0, ]

            # Get RGB image from Sentinel-2 patch
            rgb = img[:, :, [2, 1, 0]]
            rgb_rescaled = np.minimum(rgb / 0.3, 1)

            # Add everything to figure
            axs[i, 0].imshow(rgb_rescaled)
            axs[i, 1].imshow(label[:, :, 0])
            axs[i, 2].imshow(building_prob)

        plt.show()


class NumericEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model: tf.keras.Model, data_generator: DataGenerator, max_samples: int = None):
        super(NumericEvaluationCallback).__init__()
        self.model = model
        self.data_generator = data_generator
        self.max_samples = data_generator.length if max_samples is None else max_samples

    def on_epoch_end(self, epoch, logs=None):

        predictions = []
        labels = []

        def callback(img, pred, label):
            predictions.append(pred)
            labels.append(label)

        inference_loop(self.model, self.data_generator, callback, self.max_samples)
        precision = metrics.precision(labels, predictions)
        recall = metrics.recall(labels, predictions)
        f1_score = metrics.f1_score(labels, predictions)
        print(f' - f1_score: {f1_score:.3f} - precision: {precision:.3f} - recall: {recall:.3f}.')


class TensorBoardCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, log_dir: str, histogram_freq: int = 1):
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        super(TensorBoardCallback).__init__(log_dir=log_dir, histogram_freq=histogram_freq)


def inference_loop(model: tf.keras.Model, data_generator: DataGenerator, callback, max_samples: int):
    for i in range(data_generator.length):
        img, label = data_generator.get_sample(i)
        prob = model.predict(img[None, ])[0, ]
        pred = tf.math.greater(prob, tf.constant([0.2]))
        label = tf.dtypes.cast(label, tf.int8)
        pred = tf.dtypes.cast(pred, tf.int8)
        callback(img, pred, label)
        if (i + 1) == max_samples:
            break
