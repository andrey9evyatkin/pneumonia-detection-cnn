from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import math
import matplotlib.pyplot as plt

from constants import RESULT_PATH


def show_data(data, amount=5):
    normal_cases = data[data['class'] == 0].values.tolist()
    pneumonia_cases = data[data['class'] == 1].values.tolist()
    cases = normal_cases[:amount] + pneumonia_cases[:amount]
    labels = ['Normal'] * amount + ['Pneumonia'] * amount

    cases_to_display = len(cases)
    figure, axes = plt.subplots(math.ceil(cases_to_display / 5), 5, figsize=(18, 6))
    figure.suptitle('The first ' + str(amount) + ' examples of normal cases and cases of pneumonia', fontsize=18,
                    fontweight='bold')
    for index, case in enumerate(cases):
        label = labels[index]
        axis = axes[index // 5, index % 5]
        axis.imshow(case[0], cmap='gray')
        axis.set_title(label)
        axis.set_aspect('auto')
        axis.axis('off')
    figure.text(0.125, 0.05, 'Amount of normal cases: ' + str(len(normal_cases)), ha='left', va='bottom', size=15)
    figure.text(0.125, 0.01, 'Amount of pneumonia cases: ' + str(len(pneumonia_cases)), ha='left', va='bottom', size=15)
    figure.savefig(RESULT_PATH + "train_examples.png")


def show_confusion_matrix(data_labels, predicted_labels):
    matrix = confusion_matrix(data_labels, predicted_labels)
    plot_confusion_matrix(matrix, figsize=(12, 8), hide_ticks=True)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.savefig(RESULT_PATH + "confusion_matrix.png")
    confusion_matrix(data_labels, predicted_labels)
    print(confusion_matrix)
    report = classification_report(data_labels, predicted_labels, target_names=['Normal', 'Pneumonia'])
    print(report)


def show_predicted_results(data_set, predicted_labels, amount=5):
    rows = 2
    columns = 5
    f, axs = plt.subplots(rows, columns, figsize=(18, 6))
    plt.ioff()
    correctly_predicted_cases = []
    incorrectly_predicted_cases = []
    results_index = 1
    for (index, data), predicted_label in zip(data_set.iterrows(), predicted_labels):
        true_label = 'Normal' if data['class'] == 0 else 'Pneumonia'
        pred_label = 'Normal' if predicted_label == 0 else 'Pneumonia'
        if true_label == pred_label:
            correctly_predicted_cases.append((data, (true_label, pred_label)))
        else:
            incorrectly_predicted_cases.append((data, (true_label, pred_label)))
        current_row = 0 if (index // columns) % rows == 0 else 1
        ax = axs[current_row, index % columns]
        ax.imshow(data['image'], cmap='gray')
        ax.set_title('True class: ' + true_label + '\nPredicted class: ' + pred_label, pad=1.0)
        ax.set_aspect('auto')
        ax.axis('off')
        if index % (rows * columns) == 9:
            f.savefig(RESULT_PATH + '/predicted_results/' + str(results_index) + '.png')
            results_index += 1

    figure, axes = plt.subplots(rows, columns, figsize=(36, 12))
    figure.subplots_adjust(hspace=0.5)
    figure.suptitle('First ' + str(amount) + ' examples of correctly and incorrectly predicted cases',
                    fontsize=18, fontweight='bold')
    predicted_cases = correctly_predicted_cases[:amount] + incorrectly_predicted_cases[:amount]
    for index, case in enumerate(predicted_cases):
        true_label = case[1][0]
        pred_label = case[1][1]
        axis = axes[index // 5, index % 5]
        axis.imshow(case[0]['image'], cmap='gray')
        axis.set_title('True class: ' + true_label + '\nPredicted class: ' + pred_label, pad=1.0)
        axis.set_aspect('auto')
        axis.axis('off')
    figure.text(0.125, 0.05, 'Amount of correctly predicted cases: ' + str(len(correctly_predicted_cases)),
                ha='left', va='bottom', size=15)
    figure.text(0.125, 0.01, 'Amount of incorrectly predicted cases: ' + str(len(incorrectly_predicted_cases)),
                ha='left', va='bottom', size=15)
    figure.savefig(RESULT_PATH + "predicted_examples.png")


def show_feature_map_results(layers, activations):
    rows = len(layers)
    columns = 8
    figure, axes = plt.subplots(rows, columns, figsize=(36, 12))
    figure.subplots_adjust(hspace=0.5)
    figure.suptitle('Feature map results (pneumonia case)', fontsize=18, fontweight='bold')
    y_text_label = 0.89
    current_row = 0
    for layer, activation in zip(layers, activations):
        f, axs = plt.subplots(ncols=columns, figsize=(36, 12))
        number_of_activations = activation.shape[-1]
        number_of_activations = min(number_of_activations, 64)
        feature_maps_index = 1
        for column in range(number_of_activations):
            index = current_row * columns + column
            ax = axs[index % columns]
            ax.set_aspect('auto')
            ax.axis('off')
            ax.imshow(activation[0, :, :, index], cmap='viridis')
            if index % columns == 7:
                f.savefig(RESULT_PATH + '/feature_maps/' + layer + '_' + str(feature_maps_index) + '.png')
                feature_maps_index += 1
            if column < columns:
                axis = axes[current_row, column]
                axis.set_aspect('auto')
                axis.axis('off')
                axis.imshow(activation[0, :, :, index], cmap='viridis')
        figure.text(0.42, y_text_label, 'Layer: ' + layer, ha='left', va='bottom', size=15)
        y_text_label -= 0.165
        current_row += 1
    figure.savefig(RESULT_PATH + 'feature_map_results.png')


def show_result_graph(metrics):
    accuracy_metrics = [metrics['accuracy'], metrics['val_accuracy']]
    accuracy_labels = ['Train Accuracy', 'Val Accuracy']
    accuracy_graph = draw_graph(accuracy_metrics, accuracy_labels, 'Accuracy')
    accuracy_graph.savefig(RESULT_PATH + 'accuracy_graph.png')
    loss_metrics = [metrics['loss'], metrics['val_loss']]
    loss_labels = ['Train Loss', 'Val Loss']
    loss_graph = draw_graph(loss_metrics, loss_labels, 'Loss')
    loss_graph.savefig(RESULT_PATH + 'loss_graph.png')


def draw_graph(metrics, labels, y_label):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics[0], c='#0c7cba', label=labels[0])
    plt.plot(metrics[1], c='#c9062a', label=labels[1])
    plt.xticks(range(0, len(metrics[0]), 5))
    plt.xlim(0, len(metrics[0]))
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.title(labels[0] + ':' + str(metrics[0][-1]) + '\n' + labels[1] + ':' + str(metrics[1][-1]), fontsize=12)
    plt.legend()
    return plt
