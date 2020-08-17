import os 
import os.path as osp 
import logging

import coloredlogs
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score  # for accuracy metric 
from sklearn.model_selection import cross_val_predict # for cufusion matrix 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve 

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


# where to save the figures 
PROJECT_ROOT_DIR = osp.dirname(__file__)
CHAPTER_ID = "classification"
IMAGES_PATH = osp.join(PROJECT_ROOT_DIR, "image", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = osp.join(IMAGES_PATH, fig_id + "." + fig_extension)
    logging.info(f"Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)



def download_MNIST(path):
    mnist = fetch_openml('mnist_784', version=1, data_home=path)
    logging.info(mnist.keys())
    return mnist 

def imshow_MNIST(data, label): 
    serial_data = data 
    cvtImage = serial_data.reshape(28, 28)

    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(cvtImage, cmap=None)
    plt.axis("off")
    plt.title(label)
    save_fig("some_digit_plot")
#    plt.show()
#    plt.close(fig)

def plot_MNISTs(instances, images_per_row=10, **options):
    size = 28 
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    num_rows = (len(instances)-1) // images_per_row + 1   # num_rows * images_per_row = total instances 
    row_images = [] 
    num_empty = num_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * num_empty)))

    for row in range(num_rows):
        rimages = images[row * images_per_row : (row + 1 ) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)

    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(image, cmap=None, **options)
    plt.axis("off")
    save_fig("more_digits_plot")
#    plt.show() 
#    plt.close(fig)
    

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    ax.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
    ax.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")
    ax.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
    ax.plot([threshold_90_precision], [0.9], "ro")
    ax.plot([threshold_90_precision], [recall_90_precision], "ro")
    plt.legend(loc="center right", fontsize=16)   
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0 ,1])
    save_fig("precision_recall_vs_threshold_plot")
#    plt.show()

def plot_precisions_vs_recall(precisions, recalls):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(recalls, precisions, "b-", linewidth=2)
    ax.plot([0.4368, 0.4368], [0., 0.9], "r:")
    ax.plot([0.0, 0.4368], [0.9, 0.9], "r:")
    ax.plot([0.4368], [0.9], "ro")

    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    save_fig("precision_vs_recall_plot")
#    plt.show()

def plot_confusion_matrix(matrix, title:str):
    """ If you prefer color and a colorbar """ 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax, cmap=None)
    save_fig(title, tight_layout=False)
#    plt.show()


if __name__ == "__main__":

    # === Loading MNIST === #
    download_path = osp.join(PROJECT_ROOT_DIR, "..")
    mnist_data = download_MNIST(path=download_path)

    input_data, label = mnist_data["data"], mnist_data["target"]
    label = label.astype(np.uint8)  # string to integer  

    print(f"data shape {input_data.shape}")
    print(f"label shape {label.shape}")
    imshow_MNIST(data=input_data[0], label=label[0])

    example_images = input_data[:100]
    plot_MNISTs(example_images, images_per_row=10)

    input_train, input_test, label_train, label_test = input_data[:60000], input_data[60000:], label[:60000], label[60000:]

    # === Binary Classifier === #  : detector for 5 
    label_train_5 = (label_train == 5)
    label_test_5  = (label_test == 5)

    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)  # model init. 
    sgd_model.fit(input_train, label_train_5)

    number5 = input_data[0] 
    logging.info(f"model reasoning: {sgd_model.predict([number5])}")

    # === Performance Measures === # 
    """
    Accuracy 
    Confusion matrix 
    Precision and Recall 
    F1-score 
    """
    Accuracy = cross_val_score(sgd_model, input_train, label_train_5, cv=3, scoring="accuracy")  
    logging.info(f"model accuracy: {Accuracy}")

    model_pred = cross_val_predict(sgd_model, input_train, label_train_5, cv=3)
    logging.info(f"model prediction: {model_pred}")
    model_confusion = confusion_matrix(label_train_5, model_pred) 
    logging.info(f"model confusion matrix: {model_confusion}")
    perfect_pred = label_train_5 # If the model predict 100% currect... 
    perfect_confusion = confusion_matrix(label_train_5, perfect_pred)
    logging.info(f"Ideal confusion matrix: {perfect_confusion}")

    precision = precision_score(label_train_5, model_pred)
    custom_precision = model_confusion[1,1] / (model_confusion[1, 1] + model_confusion[0, 1])  # TP / (TP+FP)
    logging.info(f"precision: {precision}")
    logging.info(f"custom precision: {precision}")
    recall = recall_score(label_train_5, model_pred)
    custom_recall = model_confusion[1, 1] / (model_confusion[1, 1] + model_confusion[1,0]) # TP / (TP+FN)
    logging.info(f"recall: {recall}")
    logging.info(f"custom recall: {custom_recall}")

    F1_score = f1_score(label_train_5, model_pred)
    custom_F1_score = model_confusion[1,1] / (model_confusion[1,1] + (model_confusion[1,0]+ model_confusion[0,1])/2) # TP / (TP + (FN+FP)/2)
    logging.info(f"F1-score: {F1_score}")
    logging.info(f"custome F1-score: {custom_F1_score}")


    # === Plotting curve === # 
    pred_scores = cross_val_predict(sgd_model, input_train, label_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(label_train_5, pred_scores)

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plot_precisions_vs_recall(precisions, recalls)


    # === Multiclass Classification === # 
    """
    using SVM(support vector machine)
    """
    svm_model = SVC(gamma="auto", random_state=42)
    svm_model.fit(input_train[:1000], label_train[:1000])

    some_digit = input_data[0]
    logging.info(f"SVM reasoning: {svm_model.predict([some_digit])}")

    prediction_scores = svm_model.decision_function([some_digit])
    logging.info(f"SVM reasoning decision scores: {svm_model.decision_function([some_digit])}")
    logging.info(f"argmax index: {np.argmax(prediction_scores)}")
    logging.info(f"model prediction class types: {svm_model.classes_}")
    logging.info(f"custom SVM reasoning: {svm_model.classes_[np.argmax(prediction_scores)]}")


    # === Error Analysis === #
    """
    confusion matrix for multiple classes 
    """
    scaler = StandardScaler() 
    input_train_scaled = scaler.fit_transform(input_train.astype(np.float64))
    
    label_train_pred = cross_val_predict(sgd_model, input_train_scaled, label_train, cv=3)
    conf_mx = confusion_matrix(label_train, label_train_pred)
    logging.info(f"multipleclass conf_mx: {conf_mx}")

    plot_confusion_matrix(conf_mx, title="confusion_matrix_plot")

    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums 
    np.fill_diagonal(norm_conf_mx, 0)
    plot_confusion_matrix(norm_conf_mx, title="confusion_matrix_errors_plot")





