import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


#------------------------------------
# Visualize Train set
#------------------------------------
def visualize_train_data(train_sets):

    nb_train = len(train_sets)
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    axs = fig.subplots(1, nb_train, sharex=True)


    for i, train_set in enumerate(train_sets) :
        signal_mask = train_set["labels"] == 1
        background_mask = train_set["labels"] == 0
        axs[i].scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r")
        axs[i].scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b")
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        axs[i].set_title("Train set "+ str(i+1))
        axs[i].set_xlim([-4,8])
        axs[i].set_ylim([-4,8])
        axs[i].axhline(y=2, color='g', linestyle='--')
        axs[i].axvline(x=2, color='g', linestyle='--')
    plt.show()

#------------------------------------
# Visualize Train set
#------------------------------------
def visualize_test_data(test_sets):

    nb_test = len(test_sets)
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    axs = fig.subplots(1, nb_test, sharex=True)


    for i, test_set in enumerate(test_sets) :
        axs[i].scatter(test_set["data"]["x1"], test_set["data"]["x2"], s=10, c="black")
        axs[i].scatter(test_set["data"]["x1"], test_set["data"]["x2"], s=10,c="black")
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        axs[i].set_title("Test set "+ str(i+1))
        axs[i].set_xlim([-4,8])
        axs[i].set_ylim([-4,8])
        axs[i].axhline(y=2, color='g', linestyle='--')
        axs[i].axvline(x=2, color='g', linestyle='--')
    plt.show()

#------------------------------------
# Visualize Augmented Data
#------------------------------------
def visualize_augmented_data(train_data, augmented_data):


    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    axs = fig.subplots(1, 2, sharex=True)

    names = ["Train set", "Augmented set"]
    for i, train_set in enumerate([train_data,augmented_data]) :
        signal_mask = train_set["labels"] == 1
        background_mask = train_set["labels"] == 0
        axs[i].scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r")
        axs[i].scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b")
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        axs[i].set_title(names[i])
        axs[i].set_xlim([-4,8])
        axs[i].set_ylim([-4,8])
        axs[i].axhline(y=2, color='g', linestyle='--')
        axs[i].axvline(x=2, color='g', linestyle='--')
    plt.show()


   

#------------------------------------
# Visualize Decision Boundry
#------------------------------------
def visulaize_decicion_boundary(models, data_sets, name="Train"):

    fig = plt.figure(constrained_layout=True, figsize=(12, 4))

    for index, model in enumerate(models):

        data = data_sets[index]["data"]
        labels = data_sets[index]["labels"]

        # Plot the decision boundary
        ax = plt.subplot(1, 3, index+1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        DecisionBoundaryDisplay.from_estimator(
            model.clf,
            data,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            xlabel="x1",
            ylabel="x2",
        )
        ax.scatter(
            data["x1"],
            data["x2"],
            c=labels,
            cmap="coolwarm",
            edgecolor="black",
            s=15,
        )
    plt.suptitle("Decision Boundry plots - " + name )
    _ = plt.axis("tight")
    plt.show()







# def visual_decision_bound():


#     train_d = [train_set["data"] for train_set in train_sets]
#     test_d = [test_set["data"] for test_set in test_sets]

#     train_l = [train_set["labels"] for train_set in train_sets]
#     test_l = [test_set["labels"] for test_set in test_sets]
    

#     train_data = pd.concat(train_d)
#     test_data = pd.concat(test_d)

#     train_labels = np.vstack(train_l)
#     test_labels = np.vstack(test_l)

#     #-----------------

#     #-----------------

#     fig = plt.figure(constrained_layout=True, figsize=(10, 5))
#     axs = fig.subplots(1, 2, sharex=True)
    
    
#     disp = DecisionBoundaryDisplay.from_estimator(
#         model, train_data, response_method="predict",
#         xlabel="x1", ylabel="x2",
#         alpha=0.5,
#     )
#     # disp.ax_.scatter(train_data["x1"], train_data["x2"], c=train_labels, edgecolor="k")
#     disp.plot(cmap="bwr", alpha=0.5, ax=axs[1])
#     disp.ax_.scatter(
#         train_data["x1"], 
#         train_data["x2"], 
#         c=train_labels, 
#         cmap="coolwarm", 
#         edgecolor="black"
#     )
    
#     disp.ax_.scatter(
#         test_data["x1"], 
#         test_data["x2"], 
#         c=test_labels, 
#         cmap="coolwarm", 
#         edgecolor="black"
#     )

#     plt.show()


#     #-----------------

#     fig = plt.figure(constrained_layout=True, figsize=(10, 5))
#     axs = fig.subplots(1, 2, sharex=True)

#     feature_1, feature_2 = np.meshgrid(
#         np.linspace(min(train_data["x1"].min(),test_data["x1"].min()), max(train_data["x1"].max(),test_data["x1"].max())),
#         np.linspace(min(train_data["x2"].min(),test_data["x2"].min()), max(train_data["x2"].max(),test_data["x2"].max())),
#     )

#         grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

#         display = DecisionBoundaryDisplay.from_estimator(estimator=model, X=grid, response_method="predict", xlabel="x1", ylabel="x2")
        
#     display.plot(cmap="bwr", alpha=0.5, ax=axs[0])
#     display.ax_.scatter(
#         train_data["x1"], 
#         train_data["x2"], 
#         c=train_labels, 
#         cmap="coolwarm", 
#         edgecolor="black"
#     )
#     display.plot(cmap="bwr", alpha=0.5, ax=axs[1])
#     display.ax_.scatter(
#         test_data["x1"], 
#         test_data["x2"], 
#         c=test_labels, 
#         cmap="coolwarm", 
#         edgecolor="black"
#     )
    
#     fig.suptitle("Decision boundaries")
#     axs[0].set_title("Train points")
#     axs[0].set_xlabel("x1")
#     axs[0].set_ylabel("x2")

#     axs[1].set_title("Test points")
#     axs[1].set_xlabel("x1")
#     axs[1].set_ylabel("x2")

#     plt.show()
    


# def save_plot(plot, directory, filename) :
#     plot.savefig(directory + "/" + filename)

# def train_or_test (is_train) :
#     if is_train :
#         string = "train"
#     else :
#         string = "test"
#     return string

# def show_one_dataframe (data, ax, is_train=True) :
#     if is_train :
#         signal_mask = data["labels"] == 1
#         background_mask = data["labels"] == 0
#         ax.scatter(data[signal_mask]["x1"], data[signal_mask]["x2"], s=10, c="r")
#         ax.scatter(data[background_mask]["x1"],data[background_mask]["x2"],s=10,c="b")
#     else :
#         ax.scatter(data["x1"], data["x2"], s=10, c="black")
    
    
# def show_train_test (data_train, data_test, save_figure=False):
#     # data_train and data_test have to be given as lists even if they contain only one dataframe
#     nb_train, nb_test = len(data_train), len(data_test)
    
#     fig = plt.figure(constrained_layout=True, figsize=(15, 10))
#     axs = fig.subplots(2, nb_train, sharex=True)
    
#     for i,piece_of_data in enumerate(data_train) :
#         show_one_dataframe(piece_of_data, axs[0][i], is_train=True)
#         axs[0][i].set_xlabel("x1")
#         axs[0][i].set_ylabel("x2")
#         axs[0][i].set_title(str(i) +"th train dataframe")
#     for i,piece_of_data in enumerate(data_test) :
#         show_one_dataframe(piece_of_data, axs[1][i], is_train=False)
#         axs[1][i].set_xlabel("x1")
#         axs[1][i].set_ylabel("x2")
#         axs[1][i].set_title(str(i) +"th test dataframe")
    
#     if save_figure :
#         save_plot(fig,"visualization_output","train_test_plots.png")


# def show_superposed (data_list, is_train=True, save_figure=False) :
#     fig = plt.figure(constrained_layout=True, figsize=(5, 5))
#     ax = fig.subplots(1, 1, sharex=True)
#     for piece_of_data in data_list :
#         show_one_dataframe(piece_of_data, ax, is_train=True)
    
#     ax.set_xlabel("x1")
#     ax.set_ylabel("x2")
    
#     string = train_or_test(is_train)
#     ax.set_title("All " + string + " points")
    
#     if save_figure :
#         save_plot(fig,"visualization_output","superposed_"+string+"_plot.png")



# def show_boundary_decisions(model,train_data,test_data,save_figure=False) :

#     fig = plt.figure(constrained_layout=True, figsize=(10, 5))
#     axs = fig.subplots(1, 2, sharex=True)

#     feature_1, feature_2 = np.meshgrid(
#         np.linspace(min(train_data["x1"].min(),test_data["x1"].min()), max(train_data["x1"].max(),test_data["x1"].max())),
#         np.linspace(min(train_data["x2"].min(),test_data["x2"].min()), max(train_data["x2"].max(),test_data["x2"].max())),
#     )
#     grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

#     display = DecisionBoundaryDisplay.from_estimator(estimator=model, X=grid, response_method="predict", xlabel="x1", ylabel="x2")
    
#     display.plot(cmap="bwr", alpha=0.5, ax=axs[0])
#     display.ax_.scatter(
#         train_data["x1"], train_data["x2"], c=train_data["y"], cmap="coolwarm", edgecolor="black"
#     )
#     display.plot(cmap="bwr", alpha=0.5, ax=axs[1])
#     display.ax_.scatter(
#         test_data["x1"], test_data["x2"], c=test_data["y"], cmap="coolwarm", edgecolor="black"
#     )
    
#     fig.suptitle("Decision boundaries")
#     axs[0].set_title("Train points")
#     axs[1].set_title("Test points")
#     axs[0].set_xlabel("x1")
#     axs[0].set_ylabel("x2")
#     axs[1].set_xlabel("x1")
#     axs[1].set_ylabel("x2")
    
#     if save_figure :
#         save_plot(fig,"visualization_output","boundaries_plot.png")
    


