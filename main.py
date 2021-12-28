import tensorflow as tf
import Model
import Data
import json
import random


def main():
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/board")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training/cp.ckpt",
                                                             save_weights_only=True,
                                                             verbose=1)

    def scheduler(epoch, lr):
        """
        This function keeps the initial learning rate for the first ten epochs
        and decreases it exponentially after that.
        :param epoch:
        :param lr:
        :return:
        """
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    file_path = "./data/sudoku.csv"
    dp = Data.DataProccess(file_path)

    x_train, x_test, y_train, y_test = dp.get_data()
    lr = .001

    model = Model.get_compiled_model(lr=lr)

    batch_size = 32
    epochs = 25
    history = model.fit(x_train[:], y_train[:],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard_callback, checkpoint_callback, lr_callback])

    hd = {'epochs': epochs,
          'loss': [float(i) for i in history.history['loss']],
          'val_loss': [float(i) for i in history.history['val_loss']],
          'val_accuracy': [float(i) for i in history.history['val_accuracy']],
          'accuracy': [float(i) for i in history.history['accuracy']],
          'sc_accuracy': [float(i) for i in history.history['sparse_categorical_accuracy']],
          }

    print("history data:\n", hd)

    # Write history data to file for later plots
    with open(f"./data/{model.optimizer._name}_batch_{batch_size}_history_results.json", 'w') as f:
        json.dump(hd, f)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("test loss, test acc: ", results)


if __name__ == '__main__':
    main()
