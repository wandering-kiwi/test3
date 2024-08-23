import tensorflow as tf
(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = tf.cast(x_train/255.0, tf.float32), tf.cast(x_test/255.0, tf.float32)
y_train, y_test = tf.cast(y_train,tf.int64),tf.cast(y_test,tf.int64)

train = {'x': x_train, 'y': y_train}
test = {'x': x_test, 'y': y_test}

try:
    model = tf.keras.models.load_model("model.keras")
except:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(80, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.evaluate(test['x'], test['y'])
model.fit(train['x'], train['y'], batch_size=32, epochs=10)
model.evaluate(test['x'], test['y'])
model.save("model.keras")