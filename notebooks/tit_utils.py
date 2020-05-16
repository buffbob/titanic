import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder


def load_tit(path):
    """
    downloads data from kaggle stored at path = "../Data/"
    returns a tuple of our titanic datasets- (train,test)
    """
    train = pd.read_csv(path + 'tit_train.csv')
    test = pd.read_csv(path + "tit_test.csv")

    return (train, test)


def gscv_results_terse(model, params, X_train, y_train, X_test, y_test):
    '''
    clf = a classifier, params = a dict to feed to gridsearch_cv, score_list = list of evaluation metrics
    nuff said
    '''
    scores = ["accuracy"]
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(model, params, cv=10,
                           scoring=score)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set: \n{}".format(clf.best_params_))
    print('___________________________________')
    print('cv scores on the best estimator')
    scores = cross_val_score(clf.best_estimator_, X_train, y_train, scoring="accuracy", cv=10)
    print(scores)
    print('the average cv score is {:.3} with a std of {:.3}'.format(np.mean(scores), np.std(scores)))
    return clf


def print_gscv_results(model, params, X_train, y_train, X_test, y_test):
    '''
    clf = a classifier, params = a dict to feed to gridsearch_cv, score_list = list of evaluation metrics
    '''
    scores = ["accuracy"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, params, cv=5,
                           scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print('________________________________________________')
        print('best params for model are {}'.format(clf.best_params_))

    print('\n___________________________________\n')
    print('cv scores on the best estimator')
    scores = cross_val_score(clf.best_estimator_, X_train, y_train, scoring="accuracy", cv=10)
    print(scores)
    print('the average cv score is {:.2}\n\n'.format(np.mean(scores)))

    return clf


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    """
    X is a 2D dataset
    nuf said
    """

    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


# this dataset has unique cols so we will go through one by one

def pp_Embarked(df):
    """
    simply adds 'C' where missing values are present
    inplace imputation
    return df
    """
    df.Embarked.fillna("C", inplace=True)
    return df


def pp_Name(df):
    """
    extracts the title from the Name column
    returns- df with a new column named Title appended to original df
    """
    temp = df.Name.apply(lambda x: x.split(',')[1].split(".")[0].strip())
    df['Title'] = temp
    return df


def pp_Age(df):
    """
    imputes missing values of age through a groupby([Pclass,Title,isFemale])
    returns df with new column named Age_nonull appended to it
    """
    transformed_Age = df.groupby(["Title", 'Pclass', "Sex"])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age_nonull'] = transformed_Age
    return df


def pp_Fare(df):
    '''
    This will clip outliers to the middle 98% of the range
    '''
    temp = df['Fare'].copy()
    limits = np.percentile(temp, [1, 99])
    df.Fare = np.clip(temp, limits[0], limits[1])
    return df


def pp_AgeBin(df):
    """
    takes Age_nonull and puts in bins
    returns df with new column- AgeBin
    """
    z = df.Age_nonull.round()  # some values went to 0 so clip to 1
    binborders = np.linspace(0, 80, 17)
    z = z.clip(1, None)
    z = z.astype("int32")
    df['AgeBin'] = pd.cut(z, bins=binborders, labels=False)
    return df


def pp_Sex(df):
    """
    maps male and female to 0 and 1
    returns the df with is_Female added
    """
    df['is_Female'] = df.Sex.apply(lambda row: 0 if row == "male" else 1)  # one way
    return df


def pp_Cabin(df):
    """
    extracts the deck from the cabin. Mostly 1st class has cabin assignments. Replace
    nan with "unk". Leaves as an ordinal categorical. can be onehoted later.
    returns the df with Deck added as a column
    """
    df["Deck"] = "UNK"
    temp = df.loc[df.Cabin.notnull(), :].copy()
    temp['D'] = temp.Cabin.apply(lambda z: z[0])
    df.iloc[temp.index, -1] = temp["D"]
    # df.where(df.Deck != "0", "UNK")
    return df


def oneHot(df,col_list):
    for col in col_list: 
        newcol_names = []
        oh = OneHotEncoder(dtype="uint8",categories='auto')
    # must convert df/series to array for onehot
        vals = df[[col]].values
        temp = oh.fit_transform(vals).toarray()#converts sparse to normal array
        # the new names for columns
        for name in oh.categories_[0]:
            newcol_names.append(col + "_" + str(name))
        tempdf = pd.DataFrame(temp, columns = newcol_names)
        df = pd.concat([df, tempdf], axis=1)
    return df



def scaleNumeric(df, cols):
    """
    Standardize features by removing the mean and scaling to unit variance
    """
    ss = StandardScaler()
    scaled_features = ss.fit_transform(df[cols].values)
    for i, col in enumerate(cols):
        df[col + "_scaled"] = scaled_features[:, i]
    return df


def chooseFeatures(df, alist):
    """
    df is our dataframe with all new features added
    alist is a list of cols to select for a new dataframe
    returns  df[alist]
    """
    return df[alist]


def test_dtc(alist, df, labels):
    """
    tests a decision tree model for classification
    prints out way to much stuff
    returns a GridSearchCV classifier

    """
    a = df[alist]  # select columns
    X_train, X_test, y_train, y_test = train_test_split(a, labels, test_size=0.2, random_state=42)
    dtc = DecisionTreeClassifier()
    dtc_dict = dt_dict = [{"max_depth": [2, 5, 8, 12, 15], "min_samples_leaf": [1, 2, 3],
                           "max_features": [None, 1.0, 2, 'sqrt', X_train.shape[1]]}]
    clf = gscv_results_terse(dtc, dtc_dict, X_train, y_train, X_test, y_test)

    return clf


#########################################################
# some utilities functions to aid in ml in general

def lin_to_log_even(min_num, max_num, num_pts=10):
    """
    This really only needed in min_num << 1 and min_max >> 1
    creates an evenly spaced log space from min_num to max_num
    """
    lmin = np.log10(min_num)
    lmax = np.log10(max_num)
    ls = np.linspace(lmin, lmax, num_pts)
    log_spaces = np.power(10, ls)
    # print(["{:05f}".format(each) for each in log_spaces])
    return log_spaces


def lin_to_log_random(num1, num2, num_pts=10):
    """
    This really only needed in min_num << 1 and min_max >> 1
    creates an array of random selected pts of len num_pts
    each point is in the log space from min_num to max_num
    """
    ln1 = np.log10(num1)
    ln2 = np.log10(num2)
    range_bn = np.abs(ln2 - ln1)
    z = ln2 + np.random.rand(num_pts) * -range_bn
    zz = np.power(10, z)
    print(["{:05f}".format(each) for each in zz])
    return zz