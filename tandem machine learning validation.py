import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

#Model establishement and save
#training and validation for primary model
data_path_primary = '4-48h.xlsx'
X_primary = pd.read_excel(data_path_primary, 'training and validation data', index_col = 'Samples').values
Y_primary = pd.read_excel(data_path_primary,'training and validation target').values.ravel()
scaler_primary = StandardScaler()
decom_primary = PLSRegression(n_components = 2)
X_train_primary, X_test_primary, y_train_primary, y_test_primary = train_test_split(X_primary, Y_primary, test_size=0.3, random_state=948)
X_train_primary = scaler_primary.fit_transform(X_train_primary)
X_test_primary = scaler_primary.transform(X_test_primary)
X_train_primary = decom_primary.fit_transform(X_train_primary,y_train_primary)[0]
X_test_primary = decom_primary.transform(X_test_primary)
clf_primary = MLPClassifier(hidden_layer_sizes = (32,32),
                            solver = 'adam',learning_rate = 'adaptive',
                            verbose = False,random_state = 6,max_iter=3000)
clf_primary.fit(X_train_primary, y_train_primary)
print(clf_primary.score(X_test_primary,y_test_primary))

#training and validation for unmerged groups data
data_path_primary = '4-48h.xlsx'
X_allgroups = pd.read_excel(data_path_primary, 'training and validation data', index_col = 'Samples').values
Y_allgroups = pd.read_excel(data_path_primary,'all target').values.ravel()
scaler_allgroups = StandardScaler()
decom_allgroups = PLSRegression(n_components = 2)
X_train_allgroups, X_test_allgroups, y_train_allgroups, y_test_allgroups = train_test_split(X_allgroups, Y_allgroups, test_size=0.3, random_state=948)
X_train_allgroups = scaler_primary.fit_transform(X_train_allgroups)
X_test_allgroups = scaler_primary.transform(X_test_allgroups)
X_train_allgroups = decom_primary.fit_transform(X_train_allgroups,y_train_allgroups)[0]
X_test_allgroups = decom_primary.transform(X_test_allgroups)
clf_allgroups = MLPClassifier(hidden_layer_sizes = (32,32),
                            solver = 'adam',learning_rate = 'adaptive',
                            verbose = False,random_state = 6,max_iter=3000)
clf_allgroups.fit(X_train_allgroups, y_train_allgroups)
print(clf_allgroups.score(X_test_allgroups,y_test_allgroups))

#training for sencondary model to discriminate 4,8,12H groups
data_path_secondary_1 =  '4-12h.xlsx'
X_secondary_1 = pd.read_excel(data_path_secondary_1, 'training and validation data', index_col = 'Samples').values
Y_secondary_1 = pd.read_excel(data_path_secondary_1,'training and validation target').values.ravel()
print(X_secondary_1.shape,Y_secondary_1.shape)
scaler_secondary_1 = StandardScaler()
decom_secondary_1 = PLSRegression(n_components = 2)
clf_secondary_1 = MLPClassifier(hidden_layer_sizes = (32,32),
                                solver = 'adam',learning_rate = 'adaptive',
                                verbose = False,random_state = 6,max_iter=3000)
X_train_secondary_1, X_test_secondary_1, y_train_secondary_1, y_test_secondary_1 = train_test_split(X_secondary_1, Y_secondary_1, test_size=0.3, random_state=997)
X_train_secondary_1 = scaler_secondary_1.fit_transform(X_train_secondary_1)
X_test_secondary_1 = scaler_secondary_1.transform(X_test_secondary_1)
X_train_secondary_1 = decom_secondary_1.fit_transform(X_train_secondary_1,y_train_secondary_1)[0]
X_test_secondary_1 = decom_secondary_1.transform(X_test_secondary_1)
clf_secondary_1.fit(X_train_secondary_1, y_train_secondary_1)
print(clf_secondary_1.score(X_test_secondary_1,y_test_secondary_1))

#training for secondary model to discriminate 16-20H and 24-32H groups
data_path_secondary_2 = '16-32h.xlsx'
X_secondary_2 = pd.read_excel(data_path_secondary_2, 'training and validation data', index_col = 'Samples').values
Y_secondary_2 = pd.read_excel(data_path_secondary_2,'training and validation target').values.ravel()
scaler_secondary_2 = StandardScaler()
decom_secondary_2 = PLSRegression(n_components = 2)
clf_secondary_2 = MLPClassifier(hidden_layer_sizes = (32,32),
                                solver = 'adam',learning_rate = 'adaptive',
                                verbose = False,random_state = 6,max_iter=3000)
X_train_secondary_2, X_test_secondary_2, y_train_secondary_2, y_test_secondary_2 = train_test_split(X_secondary_2, Y_secondary_2,
                                                                                                  test_size=0.3, random_state=156)
X_train_secondary_2 = scaler_secondary_2.fit_transform(X_train_secondary_2)
X_test_secondary_2 = scaler_secondary_2.transform(X_test_secondary_2)
X_train_secondary_2 = decom_secondary_2.fit_transform(X_train_secondary_2,y_train_secondary_2)[0]
X_test_secondary_2 = decom_secondary_2.transform(X_test_secondary_2)
clf_secondary_2.fit(X_train_secondary_2, y_train_secondary_2)
print(clf_secondary_2.score(X_test_secondary_2,y_test_secondary_2))

#training for secondary model to discriminate 36-40H and 44-48H groups
data_path_secondary_3 = '36-48h.xlsx'
X_secondary_3 = pd.read_excel(data_path_secondary_3, 'training and validation data', index_col = 'Samples').values
Y_secondary_3 = pd.read_excel(data_path_secondary_3,'training and validation target').values.ravel()
scaler_secondary_3 = StandardScaler()
decom_secondary_3 = PLSRegression(n_components = 2)
clf_secondary_3 = MLPClassifier(hidden_layer_sizes = (32,32),
                                solver = 'adam',learning_rate = 'adaptive',
                                verbose = False,random_state = 6,max_iter=3000)
X_train_secondary_3, X_test_secondary_3, y_train_secondary_3, y_test_secondary_3 = train_test_split(X_secondary_3, Y_secondary_3,
                                                                                                   test_size=0.3, random_state=886)
X_train_secondary_3 = scaler_secondary_3.fit_transform(X_train_secondary_3)
X_test_secondary_3 = scaler_secondary_3.transform(X_test_secondary_3)
X_train_secondary_3 = decom_secondary_3.fit_transform(X_train_secondary_3,y_train_secondary_3)[0]
X_test_secondary_3 = decom_secondary_3.transform(X_test_secondary_3)
clf_secondary_3.fit(X_train_secondary_3, y_train_secondary_3)
print(clf_secondary_3.score(X_test_secondary_3,y_test_secondary_3))

#descision area view for primary models
classificationModels = [LogisticRegression(random_state = 4,multi_class='auto',solver = 'lbfgs'),
                       svm.SVC(C = 1.5, random_state = 4,gamma = 'auto'),
                       RandomForestClassifier(n_estimators = 128),
                       MLPClassifier(hidden_layer_sizes = (32,32),
                                     solver = 'adam',learning_rate = 'adaptive',
                                     verbose = False,random_state = 6,max_iter=3000)]
models = ['Logistic','SVM','RF','MLP']
for i in range(0,4):
    clf = classificationModels[i]
    scaler = StandardScaler()
    decom_model = PLSRegression(n_components = 2)
    X_train, X_test, y_train, y_test = train_test_split(X_primary, Y_primary, test_size=0.3, random_state=948)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_OUT = scaler.transform(X_OUT_raw)
    X_train = decom_model.fit_transform(X_train,y_train)[0]
    X_test = decom_model.transform(X_test,y_test)[0]
    # X_OUT = decom_model.transform(X_OUT,Y_OUT)[0]
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
#     print(clf.score(X_OUT,Y_OUT))
    #分类边界绘制
    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # data since we want to plot the support vectors
    model = clf.fit(X_train,y_train)

    plt.figure(dpi = 600)
    ax = plt.subplot()

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.3)
#     Y = Y.reshape(Y.shape[0])
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    y_pre = clf.predict(X_test)
    y_pre = y_pre.reshape(y_pre.shape[0])
    # y_out_pre = clf.predict(X_OUT)
#     y_out_pre = y_out_pre.reshape(y_out_pre.shape[0])
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=80, edgecolors='k',)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = plt.cm.coolwarm, s=90, edgecolors = 'k',linewidth = 0.2,marker = '^')
    ax.scatter(X_test[:,0], X_test[:,1], c=y_pre, cmap = plt.cm.coolwarm, s=20,linewidth  = 0.2, marker = '*')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = Y_OUT,cmap = plt.cm.coolwarm, s = 80, edgecolors = 'MediumSeaGreen')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = y_out_pre,cmap = plt.cm.coolwarm, s = 20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('feature 1',font = 'Arial')
    ax.set_ylabel('feature 2',font = 'Arial')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig('classification for primary %s'%models[i])
    plt.show()

#decision area view for models discriminate all groups
classificationModels = [LogisticRegression(random_state = 4,multi_class='auto',solver = 'lbfgs'),
                       svm.SVC(C = 1.5, random_state = 4,gamma = 'auto'),
                       RandomForestClassifier(n_estimators = 128),
                       MLPClassifier(hidden_layer_sizes = (32,32),
                                     solver = 'adam',learning_rate = 'adaptive',
                                     verbose = False,random_state = 6,max_iter=3000)]
models = ['Logistic','SVM','RF','MLP']
for i in range(0,4):
    clf = classificationModels[i]
    scaler = StandardScaler()
    decom_model = PLSRegression(n_components = 2)
    X_train, X_test, y_train, y_test = train_test_split(X_allgroups, Y_allgroups, test_size=0.3, random_state=948)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_OUT = scaler.transform(X_OUT_raw)
    X_train = decom_model.fit_transform(X_train,y_train)[0]
    X_test = decom_model.transform(X_test,y_test)[0]
    # X_OUT = decom_model.transform(X_OUT,Y_OUT)[0]
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
#     print(clf.score(X_OUT,Y_OUT))
    #分类边界绘制
    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # data since we want to plot the support vectors
    model = clf.fit(X_train,y_train)

    plt.figure(dpi = 600)
    ax = plt.subplot()

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.3)
#     Y = Y.reshape(Y.shape[0])
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    y_pre = clf.predict(X_test)
    y_pre = y_pre.reshape(y_pre.shape[0])
    # y_out_pre = clf.predict(X_OUT)
#     y_out_pre = y_out_pre.reshape(y_out_pre.shape[0])
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=80, edgecolors='k',)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = plt.cm.coolwarm, s=90, edgecolors = 'k',linewidth = 0.2,marker = '^')
    ax.scatter(X_test[:,0], X_test[:,1], c=y_pre, cmap = plt.cm.coolwarm, s=20,linewidth  = 0.2, marker = '*')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = Y_OUT,cmap = plt.cm.coolwarm, s = 80, edgecolors = 'MediumSeaGreen')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = y_out_pre,cmap = plt.cm.coolwarm, s = 20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('feature 1',font = 'Arial')
    ax.set_ylabel('feature 2',font = 'Arial')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig('classification for allgroups %s'%models[i])
    plt.show()

#decision area view for secodary models to discriminate 4,8,and 12h groups
classificationModels = [LogisticRegression(random_state = 4,multi_class='auto',solver = 'lbfgs'),
                       svm.SVC(C = 1.5, random_state = 4,gamma = 'auto'),
                       RandomForestClassifier(n_estimators = 128),
                       MLPClassifier(hidden_layer_sizes = (32,32),
                                     solver = 'adam',learning_rate = 'adaptive',
                                     verbose = False,random_state = 6,max_iter=3000)]
models = ['Logistic','SVM','RF','MLP']
for i in range(0,4):
    clf = classificationModels[i]
    scaler = StandardScaler()
    decom_model = PLSRegression(n_components = 2)
    X_train, X_test, y_train, y_test = train_test_split(X_secondary_1, Y_secondary_1,
                                                                        test_size=0.3, random_state=997)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_OUT = scaler.transform(X_OUT_raw)
    X_train = decom_model.fit_transform(X_train,y_train)[0]
    X_test = decom_model.transform(X_test,y_test)[0]
    # X_OUT = decom_model.transform(X_OUT,Y_OUT)[0]
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
#     print(clf.score(X_OUT,Y_OUT))

    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # data since we want to plot the support vectors
    model = clf.fit(X_train,y_train)

    plt.figure(dpi = 600)
    ax = plt.subplot()

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.3)
#     Y = Y.reshape(Y.shape[0])
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    y_pre = clf.predict(X_test)
    y_pre = y_pre.reshape(y_pre.shape[0])
    # y_out_pre = clf.predict(X_OUT)
#     y_out_pre = y_out_pre.reshape(y_out_pre.shape[0])
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=80, edgecolors='k',)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = plt.cm.coolwarm, s=90, edgecolors = 'k',linewidth = 0.2,marker = '^')
    ax.scatter(X_test[:,0], X_test[:,1], c=y_pre, cmap = plt.cm.coolwarm, s=20,linewidth  = 0.2, marker = '*')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = Y_OUT,cmap = plt.cm.coolwarm, s = 80, edgecolors = 'MediumSeaGreen')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = y_out_pre,cmap = plt.cm.coolwarm, s = 20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('feature 1',font = 'Arial')
    ax.set_ylabel('feature 2',font = 'Arial')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig('classification for secondary group one %s'%models[i])
    plt.show()

#decision area view for discriminate 16-20h and 24-32h groups
classificationModels = [LogisticRegression(random_state = 4,multi_class='auto',solver = 'lbfgs'),
                       svm.SVC(C = 1.5, random_state = 4,gamma = 'auto'),
                       RandomForestClassifier(n_estimators = 128),
                       MLPClassifier(hidden_layer_sizes = (32,32),
                                     solver = 'adam',learning_rate = 'adaptive',
                                     verbose = False,random_state = 6,max_iter=3000)]
models = ['Logistic','SVM','RF','MLP']
for i in range(0,4):
    clf = classificationModels[i]
    scaler = StandardScaler()
    decom_model = PLSRegression(n_components = 2)
    X_train, X_test, y_train, y_test = train_test_split(X_secondary_2, Y_secondary_2,
                                                                        test_size=0.3, random_state=156)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_OUT = scaler.transform(X_OUT_raw)
    X_train = decom_model.fit_transform(X_train,y_train)[0]
    X_test = decom_model.transform(X_test,y_test)[0]
    # X_OUT = decom_model.transform(X_OUT,Y_OUT)[0]
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
#     print(clf.score(X_OUT,Y_OUT))
    #分类边界绘制
    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # data since we want to plot the support vectors
    model = clf.fit(X_train,y_train)

    plt.figure(dpi = 600)
    ax = plt.subplot()

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.3)
#     Y = Y.reshape(Y.shape[0])
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    y_pre = clf.predict(X_test)
    y_pre = y_pre.reshape(y_pre.shape[0])
    # y_out_pre = clf.predict(X_OUT)
#     y_out_pre = y_out_pre.reshape(y_out_pre.shape[0])
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=80, edgecolors='k',)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = plt.cm.coolwarm, s=90, edgecolors = 'k',linewidth = 0.2,marker = '^')
    ax.scatter(X_test[:,0], X_test[:,1], c=y_pre, cmap = plt.cm.coolwarm, s=20,linewidth  = 0.2, marker = '*')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = Y_OUT,cmap = plt.cm.coolwarm, s = 80, edgecolors = 'MediumSeaGreen')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = y_out_pre,cmap = plt.cm.coolwarm, s = 20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('feature 1',font = 'Arial')
    ax.set_ylabel('feature 2',font = 'Arial')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig('classification for secondary group two %s'%models[i])
    plt.show()

#decision area view for discriminate 36-40h and 44-48h groups
classificationModels = [LogisticRegression(random_state = 4,multi_class='auto',solver = 'lbfgs'),
                       svm.SVC(C = 1.5, random_state = 4,gamma = 'auto'),
                       RandomForestClassifier(n_estimators = 128),
                       MLPClassifier(hidden_layer_sizes = (32,32),
                                     solver = 'adam',learning_rate = 'adaptive',
                                     verbose = False,random_state = 6,max_iter=3000)]
models = ['Logistic','SVM','RF','MLP']
for i in range(0,4):
    clf = classificationModels[i]
    scaler = StandardScaler()
    decom_model = PLSRegression(n_components = 2)
    X_train, X_test, y_train, y_test = train_test_split(X_secondary_3, Y_secondary_3,
                                                                        test_size=0.3, random_state=886)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_OUT = scaler.transform(X_OUT_raw)
    X_train = decom_model.fit_transform(X_train,y_train)[0]
    X_test = decom_model.transform(X_test,y_test)[0]
    # X_OUT = decom_model.transform(X_OUT,Y_OUT)[0]
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
#     print(clf.score(X_OUT,Y_OUT))
    #分类边界绘制
    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # data since we want to plot the support vectors
    model = clf.fit(X_train,y_train)

    plt.figure(dpi = 600)
    ax = plt.subplot()

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.3)
#     Y = Y.reshape(Y.shape[0])
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    y_pre = clf.predict(X_test)
    y_pre = y_pre.reshape(y_pre.shape[0])
    # y_out_pre = clf.predict(X_OUT)
#     y_out_pre = y_out_pre.reshape(y_out_pre.shape[0])
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=80, edgecolors='k',)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = plt.cm.coolwarm, s=90, edgecolors = 'k',linewidth = 0.2,marker = '^')
    ax.scatter(X_test[:,0], X_test[:,1], c=y_pre, cmap = plt.cm.coolwarm, s=20,linewidth  = 0.2, marker = '*')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = Y_OUT,cmap = plt.cm.coolwarm, s = 80, edgecolors = 'MediumSeaGreen')
    # ax.scatter(X_OUT[:,0],X_OUT[:,1],c = y_out_pre,cmap = plt.cm.coolwarm, s = 20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('feature 1',font = 'Arial')
    ax.set_ylabel('feature 2',font = 'Arial')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig('classification for secondary group three %s'%models[i])
    plt.show()

#tandem model internal validation
#data iuput
data_path_primary = '4-48h.xlsx'
X = pd.read_excel(data_path_primary, 'training and validation data', index_col = 'Samples').values
Y = pd.read_excel(data_path_primary,'training and validation target').values.ravel()
#define empty list
Four_to_twelve = []
Sixteen_to_thirtytwo = []
Thirtysix_to_fourtyeight  =  []
Predict_class = []
#primary model discrimination and secondary dataset generation
for i in X:
    u = i.reshape(1,-1)
    u = scaler_primary.transform(u)
    u = decom_primary.transform(u)
    prey = clf_primary.predict(u)
    Predict_class.append(prey)
    print("PTP:",prey)
    if prey == 0:
        Four_to_twelve.append(i)
    if prey == 1:
        Sixteen_to_thirtytwo.append(i)
    if prey == 2:
        Thirtysix_to_fourtyeight.append(i)

Four_to_twelve = np.array(Four_to_twelve)
Sixteen_to_thirtytwo = np.array(Sixteen_to_thirtytwo)
Thirtysix_to_fourtyeight = np.array(Thirtysix_to_fourtyeight)
Predict_class = np.array(Predict_class)

#preparing datasets for secondary model
Four_to_twelve = Four_to_twelve[:,[0,1,5,6,7,8,9,11,12,14,15,16,20,21,22,23,25,26,28,29,30,31,32,34,35,40,41,42]]
Sixteen_to_thirtytwo = Sixteen_to_thirtytwo[:,[0,1,2,3,4,5,6,7,8,9,
                                               10,11,13,15,16,20,21,22,23,24,
                                               26,27,28,29,31,32,33,34,35,36,
                                               37,39,40,41,42]]
Thirtysix_to_fourtyeight = Thirtysix_to_fourtyeight[:,[0,1,2,3,4,5,10,11,12,13,
                                                       14,15,16,17,19,20,21,23,24,25,
                                                       26,28,29,34,39,40,41,42]]
#predict the secondary target
Four_to_twelve_secondary_target = []
for i in Four_to_twelve:
    u = i.reshape(1,-1)
    u = scaler_secondary_1.transform(u)
    u = decom_secondary_1.transform(u)
    prey = clf_secondary_1.predict(u)
    Four_to_twelve_secondary_target.append(prey)
    print('STP_1:', prey)
Four_to_twelve_secondary_target = np.array(Four_to_twelve_secondary_target)
Sixteen_to_thirtytwo_secondary_target = []
for i in Sixteen_to_thirtytwo:
    u = i.reshape(1,-1)
    u = scaler_secondary_2.transform(u)
    u = decom_secondary_2.transform(u)
    prey = clf_secondary_2.predict(u)
    Sixteen_to_thirtytwo_secondary_target.append(prey)
    print('STP_2:', prey)
Sixteen_to_thirtytwo_secondary_target = np.array(Sixteen_to_thirtytwo_secondary_target)
Thirtysix_to_fourtyeight_secondary_target = []
for i in Thirtysix_to_fourtyeight:
    u = i.reshape(1,-1)
    u = scaler_secondary_3.transform(u)
    u = decom_secondary_3.transform(u)
    prey = clf_secondary_3.predict(u)
    Thirtysix_to_fourtyeight_secondary_target.append(prey)
    print('STP_3:',prey)
Thirtysix_to_fourtyeight_secondary_target = np.array(Thirtysix_to_fourtyeight_secondary_target)

with pd.ExcelWriter('internal_validaion_output_data_integration.xlsx') as writer:
    frame1 = pd.DataFrame(Four_to_twelve)
    frame1.to_excel(writer,sheet_name = 'Four_to_twelve_predict_data')
    frame2 = pd.DataFrame(Four_to_twelve_secondary_target)
    frame2.to_excel(writer,sheet_name = 'Four_to_twelve_secondary_target')
    frame3 = pd.DataFrame(Sixteen_to_thirtytwo)
    frame3.to_excel(writer,sheet_name = 'Sixteen_to_thirtytwo_predict_data')
    frame4 = pd.DataFrame(Sixteen_to_thirtytwo_secondary_target)
    frame4.to_excel(writer,sheet_name = 'Sixteen_to_thirtytwo_secondary_target')
    frame5 = pd.DataFrame(Thirtysix_to_fourtyeight)
    frame5.to_excel(writer,sheet_name = 'Thirtysix_to_fourtyeight_predict_data')
    frame6 = pd.DataFrame(Thirtysix_to_fourtyeight_secondary_target)
    frame6.to_excel(writer,sheet_name = 'Thirtysix_to_fourtyeight_secondary_target')
    frame7 = pd.DataFrame(X_primary)
    frame7.to_excel(writer,sheet_name = 'primary data')
    frame8 = pd.DataFrame(Predict_class)
    frame8.to_excel(writer, sheet_name = 'primary_predict_target')

#out validation
#data iuput
data_path_primary = '4-48h.xlsx'
X = pd.read_excel(data_path_primary, 'out-validation data', index_col = 'Samples').values
Y = pd.read_excel(data_path_primary,'out-validation target').values.ravel()
#define empty list
Four_to_twelve = []
Sixteen_to_thirtytwo = []
Thirtysix_to_fourtyeight  =  []
Predict_class = []
#primary model discrimination and secondary dataset generation
for i in X:
    u = i.reshape(1,-1)
    u = scaler_primary.transform(u)
    u = decom_primary.transform(u)
    prey = clf_primary.predict(u)
    Predict_class.append(prey)
    print("PTP:",prey)
    if prey == 0:
        Four_to_twelve.append(i)
    if prey == 1:
        Sixteen_to_thirtytwo.append(i)
    if prey == 2:
        Thirtysix_to_fourtyeight.append(i)

Four_to_twelve = np.array(Four_to_twelve)
Sixteen_to_thirtytwo = np.array(Sixteen_to_thirtytwo)
Thirtysix_to_fourtyeight = np.array(Thirtysix_to_fourtyeight)
Predict_class = np.array(Predict_class)

#preparing datasets for secondary model
Four_to_twelve = Four_to_twelve[:,[0,1,5,6,7,8,9,11,12,14,15,16,20,21,22,23,25,26,28,29,30,31,32,34,35,40,41,42]]
Sixteen_to_thirtytwo = Sixteen_to_thirtytwo[:,[0,1,2,3,4,5,6,7,8,9,
                                               10,11,13,15,16,20,21,22,23,24,
                                               26,27,28,29,31,32,33,34,35,36,
                                               37,39,40,41,42]]
Thirtysix_to_fourtyeight = Thirtysix_to_fourtyeight[:,[0,1,2,3,4,5,10,11,12,13,
                                                       14,15,16,17,19,20,21,23,24,25,
                                                       26,28,29,34,39,40,41,42]]
#predict the secondary target
Four_to_twelve_secondary_target = []
for i in Four_to_twelve:
    u = i.reshape(1,-1)
    u = scaler_secondary_1.transform(u)
    u = decom_secondary_1.transform(u)
    prey = clf_secondary_1.predict(u)
    Four_to_twelve_secondary_target.append(prey)
    print('STP_1:', prey)
Four_to_twelve_secondary_target = np.array(Four_to_twelve_secondary_target)
Sixteen_to_thirtytwo_secondary_target = []
for i in Sixteen_to_thirtytwo:
    u = i.reshape(1,-1)
    u = scaler_secondary_2.transform(u)
    u = decom_secondary_2.transform(u)
    prey = clf_secondary_2.predict(u)
    Sixteen_to_thirtytwo_secondary_target.append(prey)
    print('STP_2:', prey)
Sixteen_to_thirtytwo_secondary_target = np.array(Sixteen_to_thirtytwo_secondary_target)
Thirtysix_to_fourtyeight_secondary_target = []
for i in Thirtysix_to_fourtyeight:
    u = i.reshape(1,-1)
    u = scaler_secondary_3.transform(u)
    u = decom_secondary_3.transform(u)
    prey = clf_secondary_3.predict(u)
    Thirtysix_to_fourtyeight_secondary_target.append(prey)
    print('STP_3:',prey)
Thirtysix_to_fourtyeight_secondary_target = np.array(Thirtysix_to_fourtyeight_secondary_target)

with pd.ExcelWriter('outvalidation_output_integration.xlsx') as writer:
    frame1 = pd.DataFrame(Four_to_twelve)
    frame1.to_excel(writer,sheet_name = 'Four_to_twelve_predict_data')
    frame2 = pd.DataFrame(Four_to_twelve_secondary_target)
    frame2.to_excel(writer,sheet_name = 'Four_to_twelve_secondary_target')
    frame3 = pd.DataFrame(Sixteen_to_thirtytwo)
    frame3.to_excel(writer,sheet_name = 'Sixteen_to_thirtytwo_predict_data')
    frame4 = pd.DataFrame(Sixteen_to_thirtytwo_secondary_target)
    frame4.to_excel(writer,sheet_name = 'Sixteen_to_thirtytwo_secondary_target')
    frame5 = pd.DataFrame(Thirtysix_to_fourtyeight)
    frame5.to_excel(writer,sheet_name = 'Thirtysix_to_fourtyeight_predict_data')
    frame6 = pd.DataFrame(Thirtysix_to_fourtyeight_secondary_target)
    frame6.to_excel(writer,sheet_name = 'Thirtysix_to_fourtyeight_secondary_target')
    frame7 = pd.DataFrame(X)
    frame7.to_excel(writer,sheet_name = 'out_validation data')
    frame8 = pd.DataFrame(Predict_class)
    frame8.to_excel(writer, sheet_name = 'primary_predict_target_out')






