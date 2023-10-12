
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import utils
# from utils import amsasimov,amsasimov_x,compare_train_test,remerger
import math
import seaborn as sns # seaborn for nice plot quicker
from sklearn.metrics import roc_curve

# import systemetics


class Dataset_visualise():

    def __init__(self):
        self.dfall = None
        self.columns = self.dfall.columns



    def examine_dataset(self):
        print('[*] --- List of all features')
        self.dfall.columns

        print('[*] --- Examples of all features')
        self.dfall.head()

        print('[*] --- Description of all features')
        self.dfall.describe()

    def histogram_dataset(self,target,weights,columns):
        plt.figure()
        sns.set(rc={'figure.figsize':(26,30)})

        dfplot=pd.DataFrame(self.dfall, columns=columns)

        nbins = 50
        ax=dfplot[target==0].hist(weights=weights[target==0],figsize=(15,12),color='b',alpha=0.5,density=True, bins = nbins,label="B")
        ax=ax.flatten()[:dfplot.shape[1]] # to avoid error if holes in the grid of plots (like if 7 or 8 features)
        dfplot[target==1].hist(weights=weights[target==1],figsize=(15,12),color='r',alpha=0.5,density=True,ax=ax, bins = nbins,label="S")


        plt.legend(loc="best")
        plt.show

    def correlation_plots(self,target,columns = None):
        caption = ["Signal feature","Background feature"]
        if columns == None:
            columns  =self.columns
        for i in range(2):
            sns.set(rc={'figure.figsize':(32,28)})

            
            dfplot=pd.DataFrame(self.dfall, columns=columns)

            print (caption[i]," correlation matrix")
            corrMatrix = dfplot[target==i].corr()
            sns.heatmap(corrMatrix, annot=True)
            plt.show()


    def pair_plots(self,target,sample_size = 1000,columns = None):
        if columns == None:
            columns  =self.columns

        df_sample_S = self.dfall[target==1].sample(n=sample_size)
        df_sample_B = self.dfall[target==0].sample(n=sample_size)
        frames = [df_sample_S,df_sample_B]
        df_sample=pd.concat(frames)

        sns.set(rc={'figure.figsize':(16,14)})

        ax = sns.PairGrid(df_sample[columns],hue="Label")
        ax.map_upper(sns.scatterplot,alpha=0.3,size=0.3)
        ax.map_lower(sns.kdeplot,fill=True,levels=5,alpha=0.3)  # Change alpha value here
        ax.map_diag(sns.histplot,alpha=0.3)  # Change alpha value here
        ax.add_legend(title='Legend',labels=['Signal','Background'],fontsize=12)

        legend = ax._legend
        for lh in legend.legendHandles:
            lh.set_alpha(0.5)
            lh._sizes = [10]

        plt.rcParams['figure.facecolor'] = 'w'  # Set the figure facecolor to white
        plt.show()
        plt.close()

class Score_visualize():
    def __init__(self) -> None:
        pass

    def Z_curve(data,predictions,labels,weights): ## work in progress
        thresholds_list =  np.linspace(0,1,num=100)
        int_pred_sig = [weights[(labels == 1) & (predictions  > th_cut)].sum() for th_cut in thresholds_list]
        plt.plot(thresholds_list,int_pred_sig)
        plt.show()

    def roc_curve_(data,predictions,labels,weights,plot_label = "model",color='b',lw = 2):

        fpr,tpr,_ = roc_curve(y_true=labels, y_score=predictions,sample_weight=weights)
        plt.plot(fpr, tpr, color= color,lw=lw, label=plot_label

        plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.show()

        plt.close()

    





