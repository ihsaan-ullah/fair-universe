import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import cos,sin,radians
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_curve



def get_params(setting):

    L = setting["L"]
    bg_mu = np.array(setting["background_mu"])
    theta = setting["theta"]
    sg_mu = bg_mu + np.array([L * cos(radians(theta)), L * sin(radians(theta))])

    z_magnitude = setting["z_magnitude"]
    alpha = setting["alpha"]
    z = np.multiply([round(cos(radians(alpha)) ,2), round(sin(radians(alpha)), 2)], z_magnitude)

    scaling_factor = setting["scaling_factor"]
    case = setting["case"]

    box_center = sg_mu#setting["box_center"]
    box_l = setting["box_l"]

    train_comment = setting["train_comment"]
    test_comment = setting["test_comment"]

    return case, bg_mu, sg_mu, z, scaling_factor, train_comment, test_comment, box_center, box_l


def visulaize_box(ax, box_center, box_l):

    if box_l > 0:
        margin = .1

        box_x = [box_center[0]-box_l-margin, box_center[0]+box_l+margin]
        box_y = [box_center[1]-box_l-margin, box_center[1]+box_l+margin]
        
        width = box_x[1] - box_x[0]
        height = box_y[1] - box_y[0]

        ax.add_patch(
            patches.Rectangle(
                xy=(box_x[0], box_y[0]),  # point of origin.
                width=width, height=height, linewidth=1,
                color='green', fill=True, alpha=0.3))

def visualize_clock(ax, setting):

    case, bg_mu, sg_mu, z, sf, _, _, _, _ = get_params(setting)

    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    b_c = np.multiply(bg_mu, 2)
    s_c = np.multiply(sg_mu, 2)
    z_c = np.multiply(z, 2)



    if sf > 1:
        ax.plot(b_c[0], b_c[1], 'bo', markersize=40, alpha=0.3)
        ax.plot(s_c[0], s_c[1], 'ro', markersize=20, alpha=0.3)

    ax.plot(b_c[0], b_c[1], 'bo', markersize=20)
    ax.plot([b_c[0], s_c[0]], [b_c[1], s_c[1]], linestyle='-.', color="k", label="separation direction")
    ax.plot(s_c[0], s_c[1], 'ro', )
    ax.plot([b_c[0]-0.25, z_c[0]-0.25], [b_c[1]-0.25, z_c[1]-0.25], linestyle='-.', color="r", label="translation_direction")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Case - {}".format(case))

def visualize_train(ax, settings, train_set, comment=True, xy_limit=None):

    _, bg_mu, sg_mu, _, _, train_comment, _, box_center, box_l = get_params(settings)


    visulaize_box(ax, box_center, box_l)

    limit = [-8,8]
    if xy_limit is not None:
        limit =  xy_limit

    signal_mask = train_set["labels"] == 1
    background_mask = train_set["labels"] == 0
    ax.scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b", alpha=0.7, label="Background")
    ax.scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r", alpha=0.7, label="Signal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(limit)
    ax.set_ylim(limit)
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

def visualize_augmented(ax, settings, train_set, comment=True, xy_limit=None):

    _, bg_mu, sg_mu, _, _, train_comment, _, _, _ = get_params(settings)

    limit = [-8,8]
    if xy_limit is not None:
        limit =  xy_limit

    signal_mask = train_set["labels"] == 1
    background_mask = train_set["labels"] == 0
    ax.scatter(train_set["data"][background_mask]["x1"],train_set["data"][background_mask]["x2"], s=10,c="b", alpha=0.7, label="Background")
    ax.scatter(train_set["data"][signal_mask]["x1"], train_set["data"][signal_mask]["x2"], s=10, c="r", alpha=0.7, label="Signal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.axhline(y=0, color='g', linestyle='-.')
    ax.axvline(x=0, color='g', linestyle='-.')
    # ax.plot(bg_mu[0], bg_mu[1], marker="x", markersize=10, color="k", label="bg center")
    # ax.plot(sg_mu[0], sg_mu[1], marker="x", markersize=10, color="k", label="sg center")
    # ax.plot([bg_mu[0],sg_mu[0]],[bg_mu[1], sg_mu[1]], "--+", markersize=10, color="k", label="separation direction")
    ax.legend()

    if comment:
        ax.set_title("Augmented set\n" +train_comment)
    else:
        ax.set_title("Augmented set")

def visualize_test(ax, settings, test_set):

    _, bg_mu, sg_mu, z, sf, _, test_comment, box_center, box_l = get_params(settings)

    visulaize_box(ax, box_center, box_l)

    signal_mask = test_set["labels"] == 1
    background_mask = test_set["labels"] == 0




    bg_c , sg_c = [], []
    if sf > 1:
        bg_c = np.mean(test_set["data"][background_mask])
        sg_c = np.mean(test_set["data"][signal_mask])
    else:
        bg_c = [bg_mu[0]+z[0], bg_mu[1]+z[1]]
        sg_c = [sg_mu[0]+z[0], sg_mu[1]+z[1]]




    sg_data = test_set["data"][signal_mask]
    bg_data = test_set["data"][background_mask]

   
    test_set["data"][background_mask]



    ax.scatter(bg_data["x1"],bg_data["x2"], s=10,c="b", alpha=0.7, label="Background")
    ax.scatter(sg_data["x1"], sg_data["x2"], s=10, c="r", alpha=0.7, label="Signal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    ax.axhline(y=0, color='g', linestyle='-.')
    ax.axvline(x=0, color='g', linestyle='-.')
    ax.plot(bg_c[0], bg_c[1], marker="x", markersize=10, color="k", label="bg center")
    ax.plot(sg_c[0], sg_c[1], marker="x", markersize=10, color="k", label="sg center")
    ax.plot([bg_c[0],sg_c[0]],[bg_c[1], sg_c[1]], "--+", markersize=10, color="k", label="separation direction")
    ax.legend()
    ax.set_title("Test set\n" +test_comment)

    if z[0] == 0 and z[1] == 0:
        pass
    elif z[0] == 0:
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

def visualize_augmented_data(settings, train_set, augmented_set, augment_limit=None):

    fig = plt.figure(constrained_layout=True, figsize=(12, 4.5))
    axs = fig.subplots(1, 3, sharex=True)

    # Clock
    visualize_clock(axs[0],settings)
    # train
    visualize_train(axs[1], settings, train_set, comment=False, xy_limit=augment_limit)
    # visualize_augmented
    visualize_augmented(axs[2], settings, augmented_set, comment=False, xy_limit=augment_limit)
    plt.show()

def visualize_decision(ax, title, model):

    grid_resolution=100
    eps=.02

    x0_min, x0_max = (-8 - eps), (8+ eps)
    x1_min, x1_max = (-8 - eps), (8+ eps)
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )

    X_grid = np.c_[xx0.ravel(), xx1.ravel()]

    response = model.decision_function(X_grid)
    if model.model_name == "NB":
        response = model.clf.predict_proba(X_grid)[:, 1]
        # Transform with log
        epsilon = np.finfo(float).eps
        response = -np.log((1/(response+epsilon))-1)
    else:
        response = model.decision_function(X_grid)

    

    response=response.reshape(xx0.shape)


    min = np.abs(np.min(response))
    max = np.abs(np.max(response))
    max_max = np.max([min,max])
    response[0][0] = -max_max
    response[0][1] = max_max

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


        # decision boundary
        ax = plt.subplot(1, 4, 2)
        visualize_decision(ax, "Decision Boundary", model)
        
      
        # train decision boundary
        ax = plt.subplot(1, 4, 3)
        visualize_decision(ax, "Train Data", model)
        visualize_scatter(ax, train_sets[index])
       
        # test decision boundary
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

def visualize_score(df_train, df_test, obc, title,  N =8):

    N = df_train.shape[0]
    score_train = df_train.avg.values
    score_test = df_test.avg.values


    std_err_train = df_train.std_err.values
    std_err_test = df_test.std_err.values
    names = df_train.index.values

    ind = np.arange(N)
    width = 0.3 

    fig_width = 2*N
    fig_height = 8
    plt.figure(figsize=(fig_width,fig_height))
    plt.figure(figsize=(13,N))
    plt.bar(ind, score_train, yerr=std_err_train, width=width, label='train')
    plt.bar(ind + width, score_test,yerr=std_err_test, width=width, label='test')
    plt.axhline(y=obc, color='r', linestyle='-.', label="OBC Score")

    plt.xlabel('Baselines')
    plt.ylabel(title)
    plt.title(title)

    plt.xticks(ind + width / 2, names)
    plt.xticks(rotation=30)

    plt.ylim(0,1)

    # plt.legend(loc='best')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_roc_curves(name, result, settings, Y_trains, Y_tests):

    for index, _ in enumerate(result["trained_models"]):


        case, _, _, _, _, _, _, _, _ = get_params(settings[index])

        

        train_auc = result["auc_trains"][index]
        test_auc = result["auc_tests"][index]
        train_bac = result["bac_trains"][index]
        test_bac = result["bac_tests"][index]


        fig = plt.figure(figsize=(10, 4))


        # Decision Function
        ax = plt.subplot(1, 2, 1)


        fpr_train, tpr_train, _ = roc_curve(Y_trains[index],  result["Y_hat_score_trains"][index])
        fpr_test, tpr_test, _ = roc_curve(Y_tests[index],  result["Y_hat_score_tests"][index])
        ax.plot(fpr_train,tpr_train,label="Train, auc="+str(train_auc))
        ax.plot(fpr_test,tpr_test,label="Test, auc="+str(test_auc))
        ax.set_title("Decision Function ROC")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()

        # Predictions
        ax = plt.subplot(1, 2, 2)
        fpr_train, tpr_train, _ = roc_curve(Y_trains[index],  result["Y_hat_trains"][index])
        fpr_test, tpr_test, _ = roc_curve(Y_tests[index],  result["Y_hat_tests"][index])
        ax.plot(fpr_train,tpr_train,label="Train, bac="+str(train_bac))
        ax.plot(fpr_test,tpr_test,label="Test, bac="+str(test_bac))
        ax.set_title("Prediction Function ROC")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
    
        title = "Case " + str(case) + " --- " + name
        plt.suptitle(title, fontsize=15)
        plt.show()
    
    