import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin,radians
from mpl_toolkits.axes_grid1 import make_axes_locatable



def get_params(setting):

    L = setting["L"]
    bg_mu = np.array(setting["background_mu"])
    theta = setting["theta"]
    sg_mu = bg_mu + np.array([L * cos(radians(theta)), L * sin(radians(theta))])

    z_magnitude = setting["z_magnitude"]
    alpha = setting["alpha"]
    z = np.multiply([round(cos(radians(alpha)) ,2), round(sin(radians(alpha)), 2)], z_magnitude)
    case = setting["case"]

    train_comment = setting["train_comment"]
    test_comment = setting["test_comment"]

    return case, bg_mu, sg_mu, z, train_comment, test_comment

def visualize_clock(ax, setting):

    case, bg_mu, sg_mu, z, _, _ = get_params(setting)

    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    b_c = np.multiply(bg_mu, 2)
    s_c = np.multiply(sg_mu, 2)
    z_c = np.multiply(z, 2)

    ax.plot(b_c[0], b_c[1], 'bo', markersize=20)
    ax.plot([b_c[0], s_c[0]], [b_c[1], s_c[1]], linestyle='-.', color="k", label="separation direction")
    ax.plot(s_c[0], s_c[1], 'ro', )
    ax.plot([b_c[0]-0.25, z_c[0]-0.25], [b_c[1]-0.25, z_c[1]-0.25], linestyle='-.', color="r", label="translation_direction")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Case - {}".format(case))

def visualize_train(ax, settings, train_set, comment=True):

    _, bg_mu, sg_mu, _, train_comment, _ = get_params(settings)

    signal_mask = train_set["labels"] == 1
    background_mask = train_set["labels"] == 0
    ax.scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b", alpha=0.7, label="Background")
    ax.scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r", alpha=0.7, label="Signal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    ax.axhline(y=0, color='g', linestyle='-.')
    ax.axvline(x=0, color='g', linestyle='-.')
    ax.plot(bg_mu[0], bg_mu[1], marker="x", markersize=10, color="k", label="bg center")
    ax.plot(sg_mu[0], sg_mu[1], marker="x", markersize=10, color="k", label="sg center")
    ax.plot([bg_mu[0],sg_mu[0]],[bg_mu[1], sg_mu[1]], "--+", markersize=10, color="k", label="separation direction")
    ax.legend()
    if comment:
        ax.set_title("Train set\n" +train_comment)
    else:
        ax.set_title("Train set")

def visualize_test(ax, settings, test_set):

    _, bg_mu, sg_mu, z, _, test_comment = get_params(settings)

    signal_mask = test_set["labels"] == 1
    background_mask = test_set["labels"] == 0
    ax.scatter(test_set["data"][background_mask]["x1"],test_set["data"][background_mask]["x2"], s=10,c="b", alpha=0.7, label="Background")
    ax.scatter(test_set["data"][signal_mask]["x1"], test_set["data"][signal_mask]["x2"], s=10, c="r", alpha=0.7, label="Signal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    ax.axhline(y=0, color='g', linestyle='-.')
    ax.axvline(x=0, color='g', linestyle='-.')
    ax.plot(bg_mu[0]+z[0], bg_mu[1]+z[1], marker="x", markersize=10, color="k", label="bg center")
    ax.plot(sg_mu[0]+z[0], sg_mu[1]+z[1], marker="x", markersize=10, color="k", label="sg center")
    ax.plot([bg_mu[0]+z[0],sg_mu[0]+z[0]],[bg_mu[1]+z[1], sg_mu[1]+z[1]], "--+", markersize=10, color="k", label="separation direction")
    ax.legend()
    ax.set_title("Test set\n" +test_comment)


    if z[0] == 0:
        ax.axvline(x=0.25, color='r', linestyle='-.', label="translation direction")
    elif z[1] == 0:
        ax.axhline(y=0.25, color='r', linestyle='-.', label="translation direction")
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

        ax.axline((z[0], z[1]), slope=slope, linewidth=1, color='r', linestyle='-.', label="translation direction")
    ax.legend()
    ax.set_title("Test set\nz = {}\n{}".format(z, test_comment))

def visualize_augmented(ax, settins, augmented_set):

    signal_mask = augmented_set["labels"] == 1
    background_mask = augmented_set["labels"] == 0
    ax.scatter(augmented_set["data"][background_mask]["x1"],augmented_set["data"][background_mask]["x2"], s=10,c="b", label="Background")
    ax.scatter(augmented_set["data"][signal_mask]["x1"], augmented_set["data"][signal_mask]["x2"], s=10, c="r", label="Signal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Augmented set")
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    ax.axhline(y=0, color='g', linestyle='--')
    ax.axvline(x=0, color='g', linestyle='--')
    ax.legend()
    
def visualize_clocks(settings):

    fig = plt.figure(constrained_layout=True, figsize=(9, 6))
    axs = fig.subplots(2, 3, sharex=True)
    for i, ax  in enumerate(axs.flat):
        visualize_clock(ax, settings[i])
    plt.show()

def visualize_data(settings, train_set, test_set):



    fig = plt.figure(constrained_layout=True, figsize=(12, 5))
    axs = fig.subplots(1, 3, sharex=True)


    # Clock
    visualize_clock(axs[0], settings)
    # train
    visualize_train(axs[1], settings, train_set)
    # test
    visualize_test(axs[2], settings, test_set)
    plt.show()

def visualize_augmented_data(settings, train_set, augmented_set):

    fig = plt.figure(constrained_layout=True, figsize=(12, 4))
    axs = fig.subplots(1, 3, sharex=True)

    # Clock
    visualize_clock(axs[0],settings)
    # train
    visualize_train(axs[1], settings, train_set, comment=False)
    # visualize_augmented
    visualize_train(axs[2], settings, augmented_set)
    plt.show()

def visualize_decision(ax, title, model):

    grid_resolution=100
    eps=.02
    plot_method="contourf"


    x0_min, x0_max = (-8 - eps), (8+ eps)
    x1_min, x1_max = (-8 - eps), (8+ eps)
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )

    X_grid = np.c_[xx0.ravel(), xx1.ravel()]

    if model.model_name == "NB":
        response = model.clf.predict_proba(X_grid)[:, 1]
        # Transform with log
        epsilon = 0.001
        response = -np.log((1/response+epsilon)-1)
    else:
        response = model.clf.decision_function(X_grid)

    

    response=response.reshape(xx0.shape)

    ax.set_title(title)
    # plot_func = getattr(ax, plot_method)
    # surface_ = plot_func(xx0, xx1, response, 20, cmap=plt.cm.RdBu, alpha=0.5)
    im = plt.imshow(response, extent=[-8, 8, -8, 8], origin='lower', cmap="RdBu_r", alpha=0.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    ax.axhline(y=0, color='g', linestyle='--')
    ax.axvline(x=0, color='g', linestyle='--')

def visualize_scatter(ax, data_set):

    data = data_set["data"]
    labels = data_set["labels"]

    signal_mask = labels == 1
    background_mask = labels == 0

    ax.scatter(data[background_mask]["x1"],data[background_mask]["x2"], c='b', edgecolors="k")
    ax.scatter(data[signal_mask]["x1"],data[signal_mask]["x2"], c='r', edgecolors="k")

def visualize_decicion_boundary(name, settings, result, train_sets, test_sets):

    for index, model in enumerate(result["trained_models"]):

        fig = plt.figure(figsize=(30, 7))


        # Clock
        ax = plt.subplot(1, 4, 1)
        visualize_clock(ax,settings[index])


        # decision boundry
        ax = plt.subplot(1, 4, 2)
        visualize_decision(ax, "Decision Boundry", model)
        
      
        # train decision boundry
        ax = plt.subplot(1, 4, 3)
        visualize_decision(ax, "Train Data", model)
        visualize_scatter(ax, train_sets[index])
       
        # test decision boundry
        ax = plt.subplot(1, 4, 4)
        visualize_decision(ax, "Test Data", model)
        visualize_scatter(ax, test_sets[index])
       


        train_auc = round(np.mean(result["auc_trains"]),2)
        test_auc = round(np.mean(result["auc_tests"]),2)
        train_bac = round(np.mean(result["bac_trains"]),2)
        test_bac = round(np.mean(result["bac_tests"]),2)
        title = "{}\nTrain: AUC:{} BAC:{} --- Test: AUC:{} BAC:{}".format(name, train_auc, train_bac, test_auc, test_bac)
        plt.suptitle(title, fontsize=15)
        plt.show()


