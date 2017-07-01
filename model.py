'''
A tf CNN model created with Keras
'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import rmsprop


def model():
    # definition of model and its layers
    model = Sequential()

    # 2d convolutional layer with ReLU activation function - feature extraction
    model.add(Conv2D(32,
                     (5, 5),
                     input_shape=(21, 21, 3),
                     activation='relu',
                     padding='same'))

    # 2d convolutional layer with ReLU activation function - feature extraction
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))

    # pooling - downsampling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    # 2d convolutional layer with ReLU activation function - feature extraction
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    # 2d convolutional layer with ReLU activation function - feature extraction
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    # pooling - downsampling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    # flattening feature maps to a 1d vector
    model.add(Flatten())

    # dense layer (actual remembering) with dropout
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # final layer with two neurons as output
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    return model
