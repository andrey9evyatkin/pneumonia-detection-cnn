from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pickle

from constants import EPOCHS, BATCH_SIZE, RESULT_PATH
from data_preprocessing import get_dataframe
from data_visualization import *
from xray_generator import generate_data
from xray_loader import load_data


class CNN:
    def build(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block1_layer1', input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block1_layer2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block2_layer1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block2_layer2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_block3_layer1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_block3_layer2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_block3_layer3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block4_layer1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block4_layer2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block4_layer3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block5_layer1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block5_layer2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block5_layer3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # model.load_weights(RESULT_PATH + "weights.h5")
        optimizer = Adam(lr=0.0001, decay=1e-5)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, model, train_generator, val_generator):
        steps_per_epoch = train_generator.samples // BATCH_SIZE
        validation_steps = val_generator.samples // BATCH_SIZE
        stop = EarlyStopping(patience=5)
        result = model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                           validation_data=val_generator, validation_steps=validation_steps, callbacks=[stop])
        model.save_weights(RESULT_PATH + "weights.h5")
        return result

    def get_feature_map_results(self, model, case):
        layers = ['conv_block1_layer1', 'conv_block2_layer1', 'conv_block3_layer1', 'conv_block4_layer1', 'conv_block5_layer1']
        layer_outputs = [layer.output for layer in model.layers if layer.name in layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        image_array = np.expand_dims(case, axis=0) / 255.
        activations = activation_model.predict(image_array)
        return layers, activations


if __name__ == '__main__':
    train_data_set, val_data_set, test_data_set = load_data()

    train_data_set[0] = get_dataframe(train_data_set[0])
    val_data_set[0] = get_dataframe(val_data_set[0])
    test_data_set[0] = get_dataframe(test_data_set[0])

    show_data(test_data_set[0])

    train_generator = generate_data(train_data)
    val_generator = generate_data(val_data)
    test_generator = generate_data(test_data_set)

    cnn = CNN()
    model = cnn.build()
    model.summary()

    result = cnn.fit(model, train_generator, val_generator)
    with open(RESULT_PATH + 'cnn_results.bin', 'wb') as fw:
        pickle.dump(result.history, fw)

    # class Temp:
    #     pass
    # result = Temp()
    # with open(RESULT_PATH + 'cnn_results.bin', 'rb') as fr:
    #     result.history = pickle.load(fr)

    loss, score = model.evaluate(test_generator)
    print("Loss on test set: ", loss * 100)
    print("Accuracy on test set: ", score * 100)

    test_data = np.array(test_data_set[0]['image'].tolist())
    test_labels = test_generator.classes
    prediction = model.predict_classes(test_data)
    predicted_labels = prediction.reshape(1, -1)[0]

    show_result_graph(result.history)
    show_confusion_matrix(test_labels, predicted_labels)
    print("Recall score: " + str(recall_score(test_labels, predicted_labels)))
    print("Precision score: " + str(precision_score(test_labels, predicted_labels)))
    print("F1 score: " + str(f1_score(test_labels, predicted_labels)))

    show_predicted_results(test_data_set[0], predicted_labels)

    pneumonia_case = test_data_set[0][test_data_set[0]['class'] == 1]['image'].values[0]
    layer_names, image_activations = cnn.get_feature_map_results(model, pneumonia_case)
    show_feature_map_results(layer_names, image_activations)
