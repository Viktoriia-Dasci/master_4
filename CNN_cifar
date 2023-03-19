from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(num_conv_layers, num_pooling_layers, num_dense_layers, input_shape, num_classes, dropout_rate):
    model = Sequential()
    
    # Add convolutional layers
    for i in range(num_conv_layers):
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
    
    # Add pooling layers
    for i in range(num_pooling_layers):
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output for the dense layers
    model.add(Flatten())
    
    # Add dense layers
    for i in range(num_dense_layers):
        model.add(Dense(units=128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Add final output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras_tuner import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import accuracy_score

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model-building function
def build_model(hp):
    model = Sequential()

    # Add convolutional layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float('dropout_conv', 0.0, 0.5)))

    # Flatten the output for the dense layers
    model.add(Flatten())

    # Add dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(units=hp.Int('units_dense', 128, 512, 32), activation='relu'))
        model.add(Dropout(hp.Float('dropout_dense', 0.0, 0.5)))

    # Add final output layer
    model.add(Dense(units=10, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Define the Hyperband search object
tuner = Hyperband(build_model, objective='val_accuracy', max_epochs=40, factor=3, seed=1, hyperparameters=HyperParameters())

# Search for the best hyperparameters
tuner.search(X_train, y_train, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# Print the best hyperparameters found by the tuner
best_hyperparams = tuner.get_best_hyperparameters(1)[0]
print(f'Best hyperparameters: {best_hyperparams}')
