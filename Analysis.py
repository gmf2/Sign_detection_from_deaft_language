import matplotlib.pyplot as plt
import numpy as np
import util.helpers as H
import pandas as pd


#Recall per class and k
def compute_precision_recall_F1_label(predictions, labels,k,c):
    precision,recall, f1_score = 0,0,0
    # YOUR CODE HERE
    TP=0
    FP=0
    FN=0
    #TOP-3 ACCURACY
    p = H.get_ordered_predictions(predictions)
    n_samples = p.shape[0]
    # Get the top k predictions per sample.
    if k==1:
        top_k = p[:,:1]
        #top_rest=p[:,1:]
        top_rest=p[:,3:]
    if k==2:
        top_k = p[:,1:2]
        top_rest=p[:,3:]
    if k==3:
        top_k = p[:,2:3]
        top_rest=p[:,3:]
    countLab=0
    for sample_index in range(n_samples):
        #Loop per top3 in all the labels
        #TP:Predicted in k is equals to label and equals to asked class
        if labels[sample_index]==c  and top_k[sample_index]==labels[sample_index]:
            TP=TP+1
        #FP:Predicted in k is equals to label but not equals to asked class
        if top_k[sample_index]!=labels[sample_index] and labels[sample_index]==c :
            FP+=1
        #FALSE PREDICTIONS
        numFN=0
        p_rest=0
        if c==labels[sample_index]:
            if k==1:
                while p_rest<len(top_rest[sample_index]):
                    #FN
                    #print('label{}'.format(labels[sample_index]))
                    #print('top_rest[sample_index][p_rest]:{}'.format(top_rest[sample_index][p_rest]))
                    #print('p[sample_index][1]:{}'.format(p[sample_index][1]))
                    #print('p[sample_index][2]:{}'.format(p[sample_index][2]))
                    if top_rest[sample_index][p_rest]==labels[sample_index]:
                        numFN=1
                    p_rest+=1
                #if numFN==1 or p[sample_index][1]==labels[sample_index] or p[sample_index][2]==labels[sample_index]:    
                if numFN==1:
                    FN=FN+1
            if k==2:
                while p_rest<len(top_rest[sample_index]):
                    #FN
                    if top_rest[sample_index][p_rest]==labels[sample_index] :
                        numFN=1
                    p_rest+=1
                #if numFN==1 or p[sample_index][0]==labels[sample_index] or p[sample_index][2]==labels[sample_index]:
                if numFN==1:
                        FN=FN+1
            if k==3:
                while p_rest<len(top_rest[sample_index]):
                    #FN
                    if top_rest[sample_index][p_rest]==labels[sample_index]:
                        numFN=1
                    p_rest+=1
                #if numFN==1  or p[sample_index][0]==labels[sample_index] or p[sample_index][1]==labels[sample_index]:
                    if numFN==1:
                        FN=FN+1
    count=0
    for tag in labels:
        if tag==c:
            count+=1
    precision=0
    recall=0
    f1_score=0.0
    if TP!=0 or FP!=0 :
        precision=TP/(TP+FP)
    if TP!=0 or FN !=0:
        recall=TP/(TP+FN)
    if precision!=0.0 or recall !=0.0:
        print('Recall:{}'.format(recall))
        print('Precision:{}'.format(precision))
        f1_score=2*((precision*recall)/(precision+recall))
    return precision, recall, f1_score,TP,FP,FN



def plot_confusion_matrix(conf_matrix):
     # Calculate chart area size
    leftmargin = 0.5 # inches
    rightmargin = 0.5 # inches
    categorysize = 0.8 # inches
    sizelabels=18
    figwidth = leftmargin + rightmargin + (sizelabels* categorysize)           
    f = plt.figure(figsize=(figwidth, figwidth))
    labels = ["AF","HEBBEN","WAT","ZELFDE","MOETEN","AUTORIJDEN","NAAR","1","OOK","ZEGGEN","GOED","2","AANKOMEN","EERST","ZIEN","ECHT","JA","NIET"]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix, cmap=plt.get_cmap('Blues'))
    fig.colorbar(cax)
    ax.set_xticks([i for i in range(18)])
    ax.set_yticks([i for i in range(18)])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    for (i, j), z in np.ndenumerate(conf_matrix):
        ax.text(j, i, '{}'.format(z), ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


## correlation matrix
def plot_correlation_matrix(df):
    """Takes a pandas dataframe as input"""
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize = (140,100))
    plt.jet() # set the colormap to jet
    cax = ax.matshow(df.corr(), vmin=-1, vmax=1)

    ticks = list(range(len(df.columns)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(df.columns, rotation=90, horizontalalignment='left')
    ax.set_yticklabels(df.columns)
    
    fig.colorbar(cax, ticks=[-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])

    plt.tight_layout()
    plt.show()