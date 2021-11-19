import pandas as pd
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

wandb.init(project="smart_attacker", entity="unr-mpl")

def model_train_loop(train_list):
    print(f"*** Begin Training {train_list[1]} ***")
    model.fit(train_list[3], train_list[5])
    print(f"*** {train_list[1]} Trained ***")
    y_pred = train_list[2].predict(train_list[4])
    y_probas = train_list[2].predict_proba(train_list[4])

    print("*** Begin Metric Plotting ***")
    # wandb.sklearn.plot_roc(y_test, y_probas, train_list[6])
    wandb.sklearn.plot_classifier(train_list[2], train_list[3], train_list[4], train_list[5], train_list[6], y_pred,
                                  y_probas, labels, model_name=train_list[0] + "_" + train_list[1])
    print("*** Metric Plotting Completed ***")

# Get hash csv file paths
path_to_csvs = "/home/nthom/Documents/hashdata/hash_flow_csvs/"
csv_names = sorted(os.listdir(path_to_csvs), reverse=True)
csv_names_full = []
for name in csv_names:
    csv_names_full.append(path_to_csvs+name)

# Load dataset
columns = ['dim1','dim2','dim3','dim4','dim5','dim6','dim7','dim8','dim9','dim10','dim11',
         'dim12','dim13','dim14','dim15','dim16','dim17','dim18','dim19','dim20','dim21',
         'dim22','dim23','dim24','dim25','dim26','dim27','dim28','dim29','dim30','dim31','dim32','class']

for name in csv_names_full:
    print(f"*** Begin Processing {name} Dataset ***")

    dataset = read_csv(name, names=columns)
    # dataset = dataset.head(len(dataset.index)//8)

    x = dataset.drop(['class'], axis=1)
    y_original = dataset['class'].values.tolist()

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

    print("*** Dataset Loaded ***")

    umap_reducer = umap.UMAP(n_jobs=12)
    tsne_reducer = TSNE(n_jobs=12, init="pca", learning_rate="auto")

    print("*** UMAP FIT ***")
    umap_embedding = umap_reducer.fit_transform(x)
    print("UMAP Finished")
    # print("*** TSNE FIT ***")
    # tsne_embedding = tsne_reducer.fit_transform(x)
    # print("TSNE Finished")

    umap_df = pd.DataFrame(umap_embedding, columns=["dim1", "dim2"])
    # umap_df["class"] = y_original
    umap_df["class"] = dataset["class"]

    # tsne_df = pd.DataFrame(tsne_embedding, columns=["dim1", "dim2"])
    # umap_df["class"] = y_original
    # tsne_df["class"] = dataset["class"]

    # fig, ax1 = plt.subplots()
    # for index, embedding in enumerate(umap_embedding):
    #     ax1.scatter(
    #         embedding[0],
    #         embedding[1],
    #         c=y.values.tolist()[index],
    #         label=y.values.tolist()[index],
    #         s=1)
    # ax1.set_title(f'UMAP')
    # fig.show()

    fig, ax1 = plt.subplots()
    fig.set_size_inches(24, 16)
    sns.set_style("whitegrid")
    sns.scatterplot(data=umap_df, x="dim1", y="dim2", hue="class", style="class",
                    legend="full", palette=sns.color_palette("flare", as_cmap=True),
                    s=100)
    # fig.show()
    fig.savefig(f"umap_{name[46:-4]}.png", dpi=300)

    # fig, ax1 = plt.subplots()
    # ax1.scatter(
    #     tsne_embedding[:, 0],
    #     tsne_embedding[:, 1],
    #     c=y.values.tolist(),
    #     label=y.values.tolist(),
    #     s=1)
    # ax1.set_title(f'TSNE')
    # ax1.legend()
    # fig.show()

    # fig, ax1 = plt.subplots()
    # fig.set_size_inches(24, 16)
    # sns.set_style("whitegrid")
    # sns.scatterplot(data=tsne_df, x="dim1", y="dim2", hue="class", style="class",
    #                 legend="full", palette=sns.color_palette("flare", as_cmap=True),
    #                 s=100)
    # # fig.show()
    # fig.savefig(f"tsne_{name}_flows.png", dpi=300)

    #
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x.values.tolist(),
    #     y.values.tolist(),
    #     test_size=0.30)
    #
    # # Spot Check Algorithms
    # models = []
    # # models.append(('LR', LogisticRegression(n_jobs=4)))
    # # models.append(('LDA', LinearDiscriminantAnalysis()))
    # # models.append(('KNN', KNeighborsClassifier(n_jobs=4)))
    # # models.append(('CART', DecisionTreeClassifier()))
    # # models.append(('NB', GaussianNB()))
    # # models.append(('SVM', SVC(gamma='auto', max_iter=1000, probability=True)))
    # # models.append(('linearSVM', LinearSVC(max_iter=1000)))
    # # models.append(('SGD', SGDClassifier(n_jobs=12)))
    # models.append(('MLP', MLPClassifier()))
    #
    # # process_list = []
    # # for model_name, model in models:
    # #     train_list = [name, model_name, model, x_train, x_test, y_train, y_test, labels]
    # #     # process_list.append(Process(target=model_train_loop, args=(train_list,)))
    # #
    # # for p in process_list:
    # #     p.start()
    # #     p.join()
    #
    # # evaluate each model in turn
    # accuracy = []
    # precision_macro = []
    # recall_macro = []
    # f1_macro = []
    # precision_weighted = []
    # recall_weighted = []
    # f1_weighted = []
    # names = []
    # for model_name, model in models:
    #     print(f"*** Begin Training {model_name} ***")
    #     model.fit(x_train, y_train)
    #     print(f"*** {model_name} Trained ***")
    #     y_pred = model.predict(x_test)
    #     y_probas = model.predict_proba(x_test)
    #
    #     print("*** Begin Metric Plotting ***")
    #     # wandb.sklearn.plot_classifier(model, x_train, x_test, y_train, y_test, y_pred, y_probas, labels,
    #     #                               model_name=name + "_" + model_name)
    #     print("*** Metric Plotting Completed ***")
    #
    #     wandb.log({f"Accuracy": accuracy_score(y_test, y_pred)})
    #     wandb.log({f"Precision Macro": precision_score(y_test, y_pred, average='macro')})
    #     wandb.log({f"Recall Macro": recall_score(y_test, y_pred, average='macro')})
    #     wandb.log({f"F1 Macro": recall_score(y_test, y_pred, average='macro')})
    #     wandb.log({f"Precision Weighted": precision_score(y_test, y_pred, average='weighted')})
    #     wandb.log({f"Recall Weighted": recall_score(y_test, y_pred, average='weighted')})
    #     wandb.log({f"F1 Weighted": recall_score(y_test, y_pred, average='weighted')})
    #     # precision_weighted.append(precision_score(y_test, y_pred, average='weighted'))
    #     # recall_weighted.append(recall_score(y_test, y_pred, average='weighted'))
    #     # f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))
    #
    #     # accuracy.append(accuracy_score(y_test, y_pred))
    #     # precision_macro.append(precision_score(y_test, y_pred, average='macro'))
    #     # recall_macro.append(recall_score(y_test, y_pred, average='macro'))
    #     # f1_macro.append(f1_score(y_test, y_pred, average='macro'))
    #     # precision_weighted.append(precision_score(y_test, y_pred, average='weighted'))
    #     # recall_weighted.append(recall_score(y_test, y_pred, average='weighted'))
    #     # f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))
    #     # names.append(name)
    #     # result_mean = 0
    #     # for result in results:
    #     #     result_mean += result
    #     # result_mean /= len(results)
    #     # print(f'{name}, {result_mean}')
    #
    # # print(results)
    #
    # # # evaluate each model in turn
    # # results_k_fold = []
    # # names_k_fold = []
    # # for name, model in models:
    # #     kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # #     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    # #     results_k_fold.append(cv_results)
    # #     names_k_fold.append(name)
    # #     print(f'{name}: {cv_results.mean()} ({cv_results.std()})')