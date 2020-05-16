import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

"""

"""


def load_digit_data(path):
    """
    downloads data from kaggle stored a path = "../Data/"
    returns a tuple of needed data
    """
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    labels = train['label']
    train = train.drop(['label'],axis=1)
    sample_submit = pd.read_csv(path + 'sample_submission.csv')
    return train, test, labels, sample_submit


def process_digit_data(df):
    """
    df(a DataFrame) is num x 784
    scale and convert to num x 28 x 28
    return an array
    """
    df = df/255
    vals = df.values.reshape(-1,28,28,1)
    return vals
    
def encode_labels(labels):
    """
    labels(a Series) is a 1-D vector
    reshape to matrix
    onehot encode 
    """
    temp = labels.values.reshape(-1,1)
    oh = OneHotEncoder()
    newlabels = oh.fit_transform(temp)
    return newlabels.todense()
    
#####some functions for plotting images

def show_image(image_data, ax = None):
    """
    image_data = a Series- length = 764
    plots image
    returns image
    """
    image_array = image_data.values.reshape(28,28)
    if ax:
        image_plot = ax.imshow(image_array,cmap='binary')
    else:
        image_plot = plt.imshow(image_array,cmap='bone')
    return image_plot


def plot_images(images,cls_true,cls_pred=None):
    """
    images- Dataframe numx784
    cls_true- Series? len = num  // cls_pred- Series?
    
    """
    assert len(images)==len(cls_true)
    #Create figure with num X num sub-plots num = 3
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=.3,wspace=.3)
    for i,ax in enumerate(axes.flat):
        #get image
        image = images.iloc[i,:]
        z = show_image(image,ax)
        # how to display based on presence of cls_pred
        if cls_pred is None:
            xlabel="True: {}".format(cls_true[i])
        else:
            xlabel="True: {0}, Pred: {1}".format(cls_true[i],cls_pred[i])
    # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()