import pandas as pd
import sklearn
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import preprocessing
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import os
from multiprocessing import Process


# Project the data into a visualizeable space using umap and tsne
def visualizeData(inputs, dataset_name, n_neighbors, show_umap=True, show_tsne=False):
    if show_umap:
        umap_reducer = umap.UMAP(n_jobs=12, n_neighbors=n_neighbors, metric=metric)
        print("*** UMAP FIT ***")
        umap_embedding = umap_reducer.fit_transform(inputs)
        print("UMAP Finished")

        umap_df = pd.DataFrame(umap_embedding, columns=["dim1", "dim2"])
        umap_df["class"] = dataset["class"]
        print(umap_df["class"].unique())

        fig, ax1 = plt.subplots()
        fig.set_size_inches(24, 16)
        sns.set_style("whitegrid")
        sns.scatterplot(data=umap_df, x="dim1", y="dim2", s=100, legend="full", hue="class", style="class")
        # fig.show()
        fig.savefig(f"umap_{dataset_name[46:-4]}_{n_neighbors}neighbs.png", dpi=300)

    if show_tsne:
        tsne_reducer = TSNE(n_jobs=12, init="pca", learning_rate="auto")
        print("*** TSNE FIT ***")
        tsne_embedding = tsne_reducer.fit_transform(inputs)
        print("*** TSNE Finished ***")

        tsne_df = pd.DataFrame(tsne_embedding, columns=["dim1", "dim2"])
        tsne_df["class"] = dataset["class"]

        fig, ax1 = plt.subplots()
        fig.set_size_inches(24, 16)
        sns.set_style("whitegrid")
        sns.scatterplot(data=tsne_df, x="dim1", y="dim2", hue="class", style="class",
                        legend="full", palette=sns.color_palette("flare", as_cmap=True),
                        s=100)
        fig.show()
        fig.savefig(f"tsne_{dataset_name[46:-4]}.png", dpi=300)


wandb.init(project="smart_attacker", entity="unr-mpl")

# Get hash csv file paths
path_to_csvs = "/home/nthom/Documents/hashdata/hashes_cleaned/"
csv_names = sorted(os.listdir(path_to_csvs), reverse=True)
csv_names_full = []
for csv_name in csv_names:
    csv_names_full.append(path_to_csvs + csv_name)

# Load dataset
columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11',
           'dim12', 'dim13', 'dim14', 'dim15', 'dim16', 'dim17', 'dim18', 'dim19', 'dim20', 'dim21',
           'dim22', 'dim23', 'dim24', 'dim25', 'dim26', 'dim27', 'dim28', 'dim29', 'dim30', 'dim31', 'dim32', 'class']

dataset_count = 0
dataset_dict = {}
metrics_list = []
for dataset_index, dataset_name in enumerate(csv_names_full):
    print(f"*** Begin Processing {dataset_name} Dataset ***")

    dataset_dict[dataset_name] = dataset_count
    dataset_count += 1

    dataset = read_csv(dataset_name, names=columns)
    print(f"*** Parameters in {dataset_name}: {dataset.shape[0]} ***")

    # Uncomment this line to take only a portion of the data
    # dataset = dataset.head(len(dataset.index)//10)

    # x is the entire dataframe except for the class column
    x = dataset.drop(['class'], axis=1)

   # Create UMAP/TSNE visualizations
   #  visualizeData(x, dataset_name, n_neighbors=100, show_umap=True, show_tsne=False)
   #  continue

    # y_original is an unaltered list of all values in the class column
    y_original = dataset['class'].values.tolist()

    # y is a dataframe of only the class column and the values have been converted to numeric representation
    ###
    y = dataset['class']
    counter = 0

    y_temp = dataset['class'].tolist()

    for unique_value in sorted(y.unique()):
        for index, value in enumerate(y):
            if value == unique_value:
                y_temp[index] = counter
        counter += 1

    dataset["class"] = y_temp
    y = dataset['class']
    labels = dataset['class'].unique()
    ###
    print("*** Dataset Loaded ***")

    # Split the dataset into 70 percent train 30 percent test
    x_train, x_test, y_train, y_test = train_test_split(
        x.values.tolist(),
        y.values.tolist(),
        test_size=0.30)
    # Could be useful to shuffle the data, but need to make sure that class balance is maintained

    # Spot Check Algorithms
    models = []
    # models.append(('LR', LogisticRegression(n_jobs=12)))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier(n_jobs=12)))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC(gamma='auto', max_iter=1000)))
    # models.append(('linearSVM', LinearSVC()))
    # models.append(('SGD', SGDClassifier(n_jobs=12, loss="log")))
    models.append(('MLP', MLPClassifier()))


    model_dict = {}
    model_count = 0
    # evaluate each model
    for model_name, model in models:
        model_dict[model_name] = model_count
        model_count += 1
        # try:
        print(f"*** Begin Training {model_name} ***")
        model.fit(x_train, y_train)
        print(f"*** {model_name} Trained ***")

        print(f"*** Calculate Predictions and Probabilities ***")
        y_pred = model.predict(x_test)
        y_probas = model.predict_proba(x_test)
        print(f"*** Predictions and Probabilities Done ***")

        # print(f"*** Calculate 5 Fold Accuracy and F1 ***")
        # print(sorted(sklearn.metrics.SCORERS.keys()))
        # exit(0)
        # kfold = StratifiedKFold(n_splits=5, shuffle=True)
        # cv_accuracy = cross_val_score(model, x.values.tolist(), y.values.tolist(), cv=kfold, scoring='accuracy', n_jobs=12)
        # cv_f1 = cross_val_score(model, x.values.tolist(), y.values.tolist(), cv=kfold, scoring='f1_weighted', n_jobs=12)

        # cv_accuracy_average = sum(cv_accuracy)/len(cv_accuracy)
        # cv_f1_average = sum(cv_f1) / len(cv_f1)
        # print(f"*** 5 Fold Accuracy and F1 Done ***")

        print("*** Begin Metric Plotting ***")
        # wandb.sklearn.plot_classifier(model, x_train, x_test, y_train, y_test, y_pred, y_probas, labels,
        #                               model_name=dataset_name + "_" + model_name)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        wandb.log({f"{model_name} Accuracy By Num Samples": accuracy, "Num Samples": dataset.shape[0]})
        wandb.log({f"{model_name} Precision Weighted By Num Samples": precision, "Num Samples": dataset.shape[0]})
        wandb.log({f"{model_name} Recall Weighted By Num Samples": recall, "Num Samples": dataset.shape[0]})
        wandb.log({f"{model_name} F1 Weighted By Num Samples": f1, "Num Samples": dataset.shape[0]})
        # wandb.log({f"5 Fold Accuracy": cv_accuracy_average, "Dataset": dataset_name, "Model": model_name})
        # wandb.log({f"5 Fold Weighted F1": cv_f1_average, "Dataset": dataset_name, "Model": model_name})

        # metrics_list.append([dataset_name, model_name, accuracy, precision, recall, f1, cv_accuracy_average, cv_f1_average])
        metrics_list.append([dataset_dict[dataset_name], model_dict[model_name], accuracy, precision, recall, f1])
        print("*** Metric Plotting Completed ***")
        # except:
        #     print(f"{model_name} Failed")

# wandb.log({"SA_data_table": wandb.Table(data=metrics_list, columns=["dataset_name", "model_name", "accuracy",
#                                         "precision", "recall", "f1", "cv_accuracy_average", "cv_f1_average"]
#                                         )
#            }
#           )
wandb.log({"SA_data_table": wandb.Table(data=metrics_list, columns=["dataset_name", "model_name", "accuracy",
                                        "precision", "recall", "f1"]
                                        )
           }
          )