import os
import os.path as osp 
import logging 

import coloredlogs
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")

# Where to save the figures 
PROJECT_ROOT_DIR = osp.dirname(__file__)
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = osp.join(PROJECT_ROOT_DIR, "image", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = osp.join(IMAGES_PATH, fig_id + "." + fig_extension)
    logging.info(f"Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_generated_data(X, Y): 
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(X, Y, "b.")
    
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("generated_data_plot")
#    plt.show()

def plot_model_predictions(X, Y, X_new, Y_predict):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(X, Y, "b.")
    ax.plot(X_new, Y_predict, "r-")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("model_predictions_with_normal_Eq")
#    plt.show()
    
def plot_gradient_descent(weight=None, lr=0.02, weight_path=None, features=None):
    X, Y, X_b, X_new_b = features 
    
    m = len(X_b)
    plt.plot(X, Y, "b.")
    num_iter = 1000 
    
    for iter in range(num_iter):
        if iter < 10:
            Y_pred = X_new_b.dot(weight)
            style = "b-" if iter>0 else "r--"
            plt.plot(X_new, Y_pred, style)
        
        gradients = 2/m * X_b.T.dot(X_b.dot(weight) - Y)
        weight = weight - lr * gradients

        if weight_path is not None: 
            weight_path.append(weight)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\lr = {}$".format(lr), fontsize=16)



if __name__ == "__main__":

    # === Linear regression using the Normal Equation === # 
    X = 2 * np.random.rand(100, 1)
    Y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3 * x + noise
    plot_generated_data(X, Y)

    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    best_parameter = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    logging.info(f"best solution: {best_parameter}")

    """
    model inference 
    """
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    Y_predict = X_new_b.dot(best_parameter)
    logging.info(f"model prediction: {Y_predict}")
    plot_model_predictions(X, Y, X_new, Y_predict)


    # === Linear regression using batch gradient descent(GD) === # 
    lr = 0.1 
    num_iter = 1000 
    m = 100  # total_instance 

    weights = np.random.randn(2, 1) # random init. 

    for iter in range(num_iter):
        gradients = 2/m * X_b.T.dot(X_b.dot(weights) - Y) 
        weights = weights - lr * gradients
    logging.info(f"updated weights: {weights}")

    weight_path_batch_GD = [] 
    np.random.seed(42)
    weight = np.random.randn(2, 1)

    features = X, Y, X_b, X_new_b 
    plt.figure(figsize=(10,4))
    plt.subplot(131); plot_gradient_descent(weight=weight, lr=0.02, weight_path=None, features=features)
    plt.ylabel("$y$", rotation=0, fontsize=18  )
    plt.subplot(132); plot_gradient_descent(weight=weight, lr = 0.1, weight_path=weight_path_batch_GD, features=features)
    plt.subplot(133); plot_gradient_descent(weight=weight, lr=0.5, weight_path=None, features=features)

    save_fig("gradient_descent_plot")
    plt.show()
    
