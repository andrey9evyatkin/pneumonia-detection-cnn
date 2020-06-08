from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pickle

from constants import EPOCHS, BATCH_SIZE, RESULT_PATH
from data_loader import load_data, load_weights
from data_preprocessing import get_dataframe
from data_visualization import *
from xray_generator import generate_data


class CNN:
    def build(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block1_layer1', input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block1_layer2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block2_layer1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block2_layer2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='conv_block3_layer1'))
        model.add(BatchNormalization())
        model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='conv_block3_layer2'))
        model.add(BatchNormalization())
        model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='conv_block3_layer3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='conv_block4_layer1'))
        model.add(BatchNormalization())
        model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='conv_block4_layer2'))
        model.add(BatchNormalization())
        model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='conv_block4_layer3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='conv_block5_layer1'))
        model.add(BatchNormalization())
        model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='conv_block5_layer2'))
        model.add(BatchNormalization())
        model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='conv_block5_layer3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        self.initialize_layers(model)
        optimizer = Adam(learning_rate=0.0001, decay=1e-5)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def initialize_layers(self, model):
        file_weights = load_weights()
        w_1, b_1 = file_weights['block1_conv1']['block1_conv1_W_1:0'], file_weights['block1_conv1']['block1_conv1_b_1:0']
        w_2, b_2 = file_weights['block1_conv2']['block1_conv2_W_1:0'], file_weights['block1_conv2']['block1_conv2_b_1:0']
        w_4, b_4 = file_weights['block2_conv1']['block2_conv1_W_1:0'], file_weights['block2_conv1']['block2_conv1_b_1:0']
        w_5, b_5 = file_weights['block2_conv2']['block2_conv2_W_1:0'], file_weights['block2_conv2']['block2_conv2_b_1:0']
        file_weights.close()
        model.layers[1].set_weights = [w_1, b_1]
        model.layers[2].set_weights = [w_2, b_2]
        model.layers[4].set_weights = [w_4, b_4]
        model.layers[5].set_weights = [w_5, b_5]

    def fit(self, model, train_generator, val_generator):
        steps_per_epoch = train_generator.samples // BATCH_SIZE
        validation_steps = val_generator.samples // BATCH_SIZE
        stop = EarlyStopping(patience=5)
        checkpoint = ModelCheckpoint(filepath=RESULT_PATH + 'weights.h5', save_best_only=True, save_weights_only=True)
        result = model.fit(train_generator, epochs=10, steps_per_epoch=steps_per_epoch,
                           validation_data=val_generator, validation_steps=validation_steps, callbacks=[stop, checkpoint])
        self.save_results(result)
        return result

    def get_feature_map_results(self, model, image):
        layers = ['conv_block1_layer2', 'conv_block2_layer2', 'conv_block3_layer3', 'conv_block4_layer3', 'conv_block5_layer3']
        layer_outputs = [layer.output for layer in model.layers if layer.name in layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        image_array = np.expand_dims(image, axis=0)
        activations = activation_model.predict(image_array)
        return layers, activations

    def save_results(self, result):
        with open(RESULT_PATH + 'cnn_results.bin', 'wb') as fw:
            pickle.dump(result.history, fw)

    def load_results(self):
        class Temp:
            pass
        result = Temp()
        with open(RESULT_PATH + 'cnn_results.bin', 'rb') as fr:
            result.history = pickle.load(fr)
        return result


if __name__ == '__main__':
    train_data_set, val_data_set, test_data_set = load_data()

    train_data_set[0] = get_dataframe(train_data_set[0])
    val_data_set[0] = get_dataframe(val_data_set[0])
    test_data_set[0] = get_dataframe(test_data_set[0])

    show_data(train_data_set[0])

    train_generator = generate_data(train_data_set[1])
    val_generator = generate_data(val_data_set[1])
    test_generator = generate_data(test_data_set[1])

    cnn = CNN()
    model = cnn.build()
    model.summary()

    result = cnn.fit(model, train_generator, val_generator)

    pneumonia_case = test_data_set[0][test_data_set[0]['class'] == 1]['image'].values[0]
    layer_names, image_activations = cnn.get_feature_map_results(model, pneumonia_case)
    show_feature_map_results(layer_names, image_activations)

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
