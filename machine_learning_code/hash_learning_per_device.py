import pandas as pd
from pandas import read_csv
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot  as plt
import matplotlib as mpl
import wandb
import os
import numpy as np
import pickle
import umap


wandb.init(project="smart_attacker", entity="unr-mpl")

# Get hash csv file paths
path_to_csvs = "../hashes_cleaned/"
# path_to_csvs = "../hashes_cleaned_confusionMatrix/"
# path_to_csvs = "../hashes_cleaned_noRing/"
# path_to_csvs = "../hashes_cleaned_noRing_noTpBulb/"
# path_to_csvs = "../hashes_uncleaned/"

name_of_current_data = "Cleaned Hashes"
# name_of_current_data = "Uncleaned Hashes"

csv_names = sorted(os.listdir(path_to_csvs), reverse=True)
csv_names_full = []
for csv_name in csv_names:
    csv_names_full.append(path_to_csvs + csv_name)

# Load dataset
columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11',
           'dim12', 'dim13', 'dim14', 'dim15', 'dim16', 'dim17', 'dim18', 'dim19', 'dim20', 'dim21',
           'dim22', 'dim23', 'dim24', 'dim25', 'dim26', 'dim27', 'dim28', 'dim29', 'dim30', 'dim31', 'dim32', 'class']

dataset_count = 10
dataset_dict = {}
metrics_list = []
for dataset_index, dataset_name in enumerate(csv_names_full):
    print(f"*** Begin Processing {dataset_name} Dataset ***")

    dataset_dict[dataset_name] = dataset_count
    dataset_count -= 1

    if dataset_dict[dataset_name] != 10 and dataset_dict[dataset_name] != 5 and dataset_dict[dataset_name] != 1:
        continue

    dataset = read_csv(dataset_name, names=columns)
    print(f"*** Parameters in {dataset_name}: {dataset.shape[0]} ***")
    # for device_name in dataset["class"].unique():
    #     num_samples = len((dataset[dataset["class"] == device_name]).index)
    #     print(f"*** Samples for device: {device_name} in {dataset_name}: {num_samples} ({num_samples/dataset.shape[0]}%) ***")
    # continue

    # Uncomment this line to take only a portion of the data
    # dataset = dataset.head(len(dataset.index)//10)

    # x is the entire dataframe except for the class column
    x = dataset.drop(['class'], axis=1)

    # y_original is an unaltered list of all values in the class column
    y_original = dataset['class'].values.tolist()

    # y is a dataframe of only the class column and the values have been converted to numeric representation
    y = dataset['class']

    counter = 0
    y_temp = dataset['class'].tolist()
    class_label_dict = {}

    for unique_value in sorted(y.unique()):
        class_label_dict[counter] = unique_value
        for index, value in enumerate(y):
            if value == unique_value:
                y_temp[index] = counter
        counter += 1

    dataset["class"] = y_temp
    y = dataset['class']
    labels_numeric = dataset['class'].unique()

    # labels_string = []
    # for num in range(22):
    # # for num in range(21):
    # # for num in range(20):
    #     labels_string.append(class_label_dict[num])

    print("*** Dataset Loaded ***")


    umap_reducer = umap.UMAP(n_jobs=12, n_neighbors=10)

    # class1 = dataset[dataset['class'] == 7]
    # class2 = dataset[dataset['class'] == 19]
    # class_dataframes = [class1, class2]
    # class3 = pd.concat(class_dataframes, ignore_index=True)["class"]
    temp1 = dataset[dataset['class'] == 12]
    temp2 = dataset[dataset['class'] == 20]
    temp4 = dataset[dataset['class'] == 10]
    dataframes = [temp1, temp2, temp4]
    temp3 = pd.concat(dataframes, ignore_index=True)
    # temp3 = temp3.sample(frac=1)
    print("*** Start UMAP Fit ***")
    umap_embedding = umap_reducer.fit_transform(temp3.drop(['class'], axis=1).values.tolist())
    print("*** Completed UMAP Fit ***")
    umap_df = pd.DataFrame(umap_embedding, columns=["dim1", "dim2"])
    umap_df["class"] = temp3["class"]

    # fig, ax = plt.subplots()
    sns.set(style="darkgrid", context="paper", rc={'figure.figsize':(3,1.5)})
    mpl.rcParams['figure.dpi'] = 600
    plt.xlabel('UMAP Dim. 0')
    plt.ylabel('UMAP Dim. 1')
    # plt.title(f'UMAP Visualization of the {name_of_current_data} {dataset_dict[dataset_name]}-Minute Set')
    sns.scatterplot(data=umap_df, x="dim1", y="dim2", hue="class", style="class", legend=None,
                    palette=['red', 'blue', "green"], s=5)
    # plt.show()
    plt.savefig(f"{name_of_current_data}_{dataset_dict[dataset_name]}")
    continue

    x = {}
    y = {}
    x["train"] = []
    y["train"] = []

    for device_name in labels_numeric:
        # Get the part of the dataset which pertains to the current device
        temp = dataset[dataset['class'] == device_name]
        # Shuffle the part of the dataset for the current device
        temp_shuffled = temp.sample(frac=1)
        length_temp_shuffled = len(temp_shuffled.index)

        x[device_name] = {}
        y[device_name] = {}
        x[device_name]["test"] = []
        y[device_name]["test"] = []

        temp_shuffled_test = temp_shuffled[:int(length_temp_shuffled * .2)]
        temp_shuffled_train = temp_shuffled[int(length_temp_shuffled * .2):]

        x[device_name]["test"] = (x[device_name]["test"] + temp_shuffled_test.drop(['class'], axis=1).values.tolist())
        y[device_name]["test"] = (y[device_name]["test"] + temp_shuffled_test['class'].values.tolist())

        x["train"] = (x["train"] + temp_shuffled_train.drop(['class'], axis=1).values.tolist())
        y["train"] = (y["train"] + temp_shuffled_train['class'].values.tolist())

        temp2 = list(zip(x["train"], y["train"]))
        random.shuffle(temp2)
        x["train"], y["train"] = [[i for i, j in temp2], [j for i, j in temp2]]

    print(len(x[device_name]["test"]), len(y[device_name]["test"]),
          len(x["train"]), len(y["train"]))

    num_samples_per_device = []
    for device_name in labels_numeric:
        num_samples_per_device.append(len(y[device_name]["test"]))
    total_samples = sum(num_samples_per_device)

    # weight_per_device = {}
    # for dev_index, device_name in enumerate(labels_numeric):
    #     weight_per_device[class_label_dict[device_name]] = num_samples_per_device[dev_index]/total_samples
    weight_per_device = []
    for dev_index, device_name in enumerate(labels_numeric):
        weight_per_device.append(num_samples_per_device[dev_index]/total_samples)
    weight_sum = sum(weight_per_device)
    # continue
    models = []
    models.append((0, MLPClassifier()))
    print(models[0][1].get_params())
    for model_name, model in models:
        print("*** Begin Training ***")
        model.fit(x["train"], y["train"])
        print("*** Completed Training ***")

        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        for dev_index, device_name in enumerate(labels_numeric):
            y_pred = model.predict(x[device_name]["test"])

            total_accuracy += (accuracy_score(y[device_name]["test"], y_pred) * weight_per_device[dev_index])
            total_precision += (precision_score(y[device_name]["test"], y_pred, average='weighted') * weight_per_device[dev_index])
            total_recall += (recall_score(y[device_name]["test"], y_pred, average='weighted') * weight_per_device[dev_index])
            total_f1 += (f1_score(y[device_name]["test"], y_pred, average='weighted') * weight_per_device[dev_index])

            accuracy = accuracy_score(y[device_name]["test"], y_pred)
            precision = precision_score(y[device_name]["test"], y_pred, average='weighted')
            recall = recall_score(y[device_name]["test"], y_pred, average='weighted')
            f1 = f1_score(y[device_name]["test"], y_pred, average='weighted')

            wandb.log({f"{class_label_dict[device_name]} accuracy SR on {name_of_current_data}": accuracy,
                          "Dataset": dataset_dict[dataset_name],
                          "Num Samples": dataset.shape[0]})
            wandb.log({f"{class_label_dict[device_name]} precision SR on {name_of_current_data}":precision,
                          "Dataset": dataset_dict[dataset_name],
                          "Num Samples": dataset.shape[0]})
            wandb.log({f"{class_label_dict[device_name]} recall SR on {name_of_current_data}": recall,
                          "Dataset": dataset_dict[dataset_name],
                          "Num Samples": dataset.shape[0]})
            wandb.log({f"{class_label_dict[device_name]} f1 SR on {name_of_current_data}": f1,
                          "Dataset": dataset_dict[dataset_name],
                          "Num Samples": dataset.shape[0]})

        # total_accuracy /= 22
        # total_precision /= 22
        # total_recall /= 22
        # total_f1 /= 22

        wandb.log({f"Total accuracy SR on {name_of_current_data}": total_accuracy,
                      "Dataset": dataset_dict[dataset_name],
                      "Num Samples": dataset.shape[0]})
        wandb.log({f"Total precision SR on {name_of_current_data}": total_precision,
                      "Dataset": dataset_dict[dataset_name],
                      "Num Samples": dataset.shape[0]})
        wandb.log({f"Total recall SR on {name_of_current_data}": total_recall,
             "Dataset": dataset_dict[dataset_name],
             "Num Samples": dataset.shape[0]})
        wandb.log({f"Total f1 SR on {name_of_current_data}": total_f1,
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})