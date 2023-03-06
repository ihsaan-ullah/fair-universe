import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from math import cos,sin,radians

from matplotlib.colors import ListedColormap

#------------------------------------
# Visualize Drawings
#------------------------------------
def visualize_data(settings, train_set, test_set):


    train_comment = settings["train_comment"]
    test_comment = settings["test_comment"]

    L = settings["L"]
    bg_mu = np.array(settings["background_mu"])
    bg_sigma = np.array(settings["background_sigma"])
    theta = settings["theta"]
    sg_mu = bg_mu + np.array([L * cos(radians(theta)), L * sin(radians(theta))])


    # z_amplitude = settings["z_amplitude"]
    # alpha = settings["alpha"]
    # z = np.multiply([cos(radians(alpha)), sin(radians(alpha))], z_amplitude)

    z = settings["z"]
    case = settings["case"]

    


    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    axs = fig.subplots(1, 2, sharex=True)

    # Train set
    signal_mask = train_set["labels"] == 1
    background_mask = train_set["labels"] == 0
    axs[0].scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b", alpha=0.7, label="Background")
    axs[0].scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r", alpha=0.7, label="Signal")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    axs[0].set_xlim([-8,8])
    axs[0].set_ylim([-8,8])
    axs[0].axhline(y=0, color='g', linestyle='-.')
    axs[0].axvline(x=0, color='g', linestyle='-.')
    axs[0].plot(bg_mu[0], bg_mu[1], marker="x", markersize=10, color="k", label="bg center")
    axs[0].plot(sg_mu[0], sg_mu[1], marker="x", markersize=10, color="k", label="sg center")
    axs[0].plot([bg_mu[0],sg_mu[0]],[bg_mu[1], sg_mu[1]], "--+", markersize=10, color="k", label="separation direction")
    axs[0].legend()
    axs[0].set_title("Train set\n" +train_comment)

    # Test set
    signal_mask = test_set["labels"] == 1
    background_mask = test_set["labels"] == 0
    axs[1].scatter(test_set["data"][background_mask]["x1"],test_set["data"][background_mask]["x2"], s=10, c="b", alpha=0.7, label="Background")
    axs[1].scatter(test_set["data"][signal_mask]["x1"], test_set["data"][signal_mask]["x2"], s=10, c="r", alpha=0.7, label="Signal")
    
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")
    axs[1].set_ylim([-8,8])
    axs[1].set_ylim([-8,8])
    axs[1].axhline(y=0, color='g', linestyle='-.')
    axs[1].axvline(x=0, color='g', linestyle='-.')
    axs[1].plot(bg_mu[0]+z[0], bg_mu[1]+z[1], marker="x", markersize=10, color="k", label="bg center")
    axs[1].plot(sg_mu[0]+z[0], sg_mu[1]+z[1], marker="x", markersize=10, color="k", label="sg center")
    axs[1].plot([bg_mu[0]+z[0],sg_mu[0]+z[0]],[bg_mu[1]+z[1], sg_mu[1]+z[1]], "--+", markersize=10, color="k", label="separation direction")
    

    if z[0] == 0:
        axs[1].axvline(x=0.25, color='r', linestyle='-.', label="translation direction")
    elif z[1] == 0:
        axs[1].axhline(y=0.25, color='r', linestyle='-.', label="translation direction")
    else:
        slope = 0

        if (z[0] < 1) & (z[1] < 1) :
            slope = 1
        elif (z[0] > 1) & (z[1] > 1) :
            slope = 1
        elif (z[0] > 1) & (z[1] < 1) :
            slope = -1
        else:
            slope = -1

        axs[1].axline((z[0], z[1]), slope=slope, linewidth=1, color='r', linestyle='-.', label="translation direction")
    axs[1].legend()
    axs[1].set_title("Test set\nz = {}\n{}".format(z, test_comment))

    plt.suptitle("Case - {}".format(case))
    plt.show()

# #------------------------------------
# # Visualize Train set
# #------------------------------------
# def visualize_train_data(train_sets):

#     nb_train = len(train_sets)
#     fig = plt.figure(constrained_layout=True, figsize=(10, 4))
#     axs = fig.subplots(1, nb_train, sharex=True)


#     for i, train_set in enumerate(train_sets) :
#         signal_mask = train_set["labels"] == 1
#         background_mask = train_set["labels"] == 0
#         axs[i].scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r")
#         axs[i].scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b")
#         axs[i].set_xlabel("x1")
#         axs[i].set_ylabel("x2")
#         axs[i].set_title("Train set "+ str(i+1))
#         axs[i].set_xlim([-4,8])
#         axs[i].set_ylim([-4,8])
#         axs[i].axhline(y=2, color='g', linestyle='--')
#         axs[i].axvline(x=2, color='g', linestyle='--')
#     plt.show()

# #------------------------------------
# # Visualize Train set
# #------------------------------------
# def visualize_test_data(test_sets):

#     nb_test = len(test_sets)
#     fig = plt.figure(constrained_layout=True, figsize=(10, 4))
#     axs = fig.subplots(1, nb_test, sharex=True)


#     for i, test_set in enumerate(test_sets) :
#         axs[i].scatter(test_set["data"]["x1"], test_set["data"]["x2"], s=10, c="black")
#         axs[i].scatter(test_set["data"]["x1"], test_set["data"]["x2"], s=10,c="black")
#         axs[i].set_xlabel("x1")
#         axs[i].set_ylabel("x2")
#         axs[i].set_title("Test set "+ str(i+1))
#         axs[i].set_xlim([-4,8])
#         axs[i].set_ylim([-4,8])
#         axs[i].axhline(y=2, color='g', linestyle='--')
#         axs[i].axvline(x=2, color='g', linestyle='--')
#     plt.show()





#------------------------------------
# Visualize Augmented Data
#------------------------------------
def visualize_augmented_data(settings, train_data, augmented_data):

    L = settings["L"]
    bg_mu = np.array(settings["background_mu"])
    bg_sigma = np.array(settings["background_sigma"])
    theta = settings["theta"]
    sg_mu = bg_mu + np.array([L * cos(radians(theta)), L * sin(radians(theta))])


   
    case = settings["case"]


    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    axs = fig.subplots(1, 2, sharex=True)

    names = ["Train set", "Augmented set"]
    for i, train_set in enumerate([train_data,augmented_data]) :
        signal_mask = train_set["labels"] == 1
        background_mask = train_set["labels"] == 0
        axs[i].scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b", label="Background")
        axs[i].scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r", label="Signal")
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        axs[i].set_title(names[i])
        axs[i].set_xlim([-20,20])
        axs[i].set_ylim([-20,20])
        axs[i].axhline(y=0, color='g', linestyle='--')
        axs[i].axvline(x=0, color='g', linestyle='--')
        if i == 0:
            axs[i].plot(bg_mu[0], bg_mu[1], marker="x", markersize=10, color="k", label="bg center")
            axs[i].plot(sg_mu[0], sg_mu[1], marker="x", markersize=10, color="k", label="sg center")
            axs[i].plot([bg_mu[0],sg_mu[0]],[bg_mu[1], sg_mu[1]], "--+", markersize=10, color="k", label="separation direction")
        axs[i].legend()
    plt.suptitle("Case - {}".format(case))
    plt.show()


   


# def visualize_decicion_boundary_old(models, train_sets, test_sets):

    
#     for index, model in enumerate(models):

#         fig = plt.figure(figsize=(11, 5))

#         train_data = train_sets[index]["data"]
#         train_labels = train_sets[index]["labels"]

#         test_data = test_sets[index]["data"]
#         test_labels = test_sets[index]["labels"]

#         plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

#         # Plot the decision boundary
#         ax = plt.subplot(1, 2, 1)
#         DecisionBoundaryDisplay.from_estimator(
#             model.clf,
#             train_data,
#             cmap=plt.cm.RdYlBu,
#             response_method="predict",
#             ax=ax,
#             xlabel="x1",
#             ylabel="x2",
#         )
#         ax.scatter(
#             train_data["x1"],
#             train_data["x2"],
#             c=train_labels,
#             cmap="coolwarm",
#             edgecolor="black",
#             s=15,
#         )
#         ax.set_title("Train Decision Boundry")
#         ax.axhline(y=0, color='b', linestyle='--')
#         ax.axvline(x=0, color='b', linestyle='--')
#         ax.set_xlim([-15,15])
#         ax.set_ylim([-15,15])

#         ax = plt.subplot(1, 2, 2)
#         DecisionBoundaryDisplay.from_estimator(
#             model.clf,
#             test_data,
#             cmap=plt.cm.RdYlBu,
#             response_method="predict",
#             ax=ax,
#             xlabel="x1",
#             ylabel="x2",
#         )
        
#         ax.scatter(
#             test_data["x1"],
#             test_data["x2"],
#             c=test_labels,
#             cmap="coolwarm",
#             edgecolor="black",
#             s=15,
#         )
#         ax.set_title("Test Decision Boundry")
#         ax.axhline(y=0, color='b', linestyle='--')
#         ax.axvline(x=0, color='b', linestyle='--')
#         ax.set_xlim([-15,15])
#         ax.set_ylim([-15,15])


#     _ = plt.axis("tight")
#     plt.show()

#------------------------------------
# Visualize Decision Boundry
#------------------------------------
def visualize_decicion_boundary(models, train_sets, test_sets):

    
    for index, model in enumerate(models):

        fig = plt.figure(figsize=(12, 4))

        train_data = train_sets[index]["data"]
        train_labels = train_sets[index]["labels"]

        trian_signal_mask = train_labels == 1
        train_background_mask = train_labels == 0

        test_data = test_sets[index]["data"]
        test_labels = test_sets[index]["labels"]

        test_signal_mask = test_labels == 1
        test_background_mask = test_labels == 0



        

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        
    
        x_min, x_max = train_data["x1"].min() - 0.5, train_data["x1"].max() + 0.5
        y_min, y_max = train_data["x2"].min() - 0.5, train_data["x2"].max() + 0.5
        ax = plt.subplot(1, 3, 1)
        ax.set_title("Decision Boundry")
        DecisionBoundaryDisplay.from_estimator(
            model.clf, train_data, cmap=cm, alpha=0.8, ax=ax, eps=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)



        ax = plt.subplot(1, 3, 2)
        ax.set_title("Train Data")
        DecisionBoundaryDisplay.from_estimator(model.clf, train_data, cmap=cm, alpha=0.8, ax=ax, eps=0.5)
        ax.scatter(train_data[train_background_mask]["x1"],train_data[train_background_mask]["x2"], c='b', edgecolors="k", label="Background")
        ax.scatter(train_data[trian_signal_mask]["x1"],train_data[trian_signal_mask]["x2"], c='r', edgecolors="k", label="Signal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()



        x_min, x_max = test_data["x1"].min() - 0.5, test_data["x1"].max() + 0.5
        y_min, y_max = test_data["x2"].min() - 0.5, test_data["x2"].max() + 0.5
        ax = plt.subplot(1, 3, 3)
        ax.set_title("Test Data")
        DecisionBoundaryDisplay.from_estimator(model.clf, train_data, cmap=cm, alpha=0.8, ax=ax, eps=0.5)
        ax.scatter(test_data[test_background_mask]["x1"],test_data[test_background_mask]["x2"], c='b', edgecolors="k", label="Background")
        ax.scatter(test_data[test_signal_mask]["x1"],test_data[test_signal_mask]["x2"], c='r', edgecolors="k", label="Signal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()

    plt.show()


