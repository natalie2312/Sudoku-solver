import tensorflow as tf


def build_uncompiled_model():
    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(81 * 9),
        tf.keras.layers.Reshape((-1, 9)),
        tf.keras.layers.Activation('softmax')

    ])

    return model


def get_compiled_model(metrics=None, lr=.001):
    if metrics is None:
        metrics = ['accuracy',
                   tf.keras.metrics.sparse_categorical_accuracy]

    model = build_uncompiled_model()

    model.compile(tf.keras.optimizers.Adam(lr=lr, name='Adam'),
                  loss='sparse_categorical_crossentropy',
                  metrics=[i for i in metrics])

    return model
