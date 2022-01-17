import pandas as pd
from pandas import read_csv
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot  as plt
import wandb
import os
import numpy as np
import pickle


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

    labels_string = []
    for num in range(22):
    # for num in range(21):
    # for num in range(20):
        labels_string.append(class_label_dict[num])

    print("*** Dataset Loaded ***")

    x = {}
    y = {}

    for device_name in labels_numeric:
        x[device_name] = {}
        y[device_name] = {}
        temp_shuffled = pd.read_pickle(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}/"
                                f"shuffled_{name_of_current_data}-{dataset_dict[dataset_name]}_device-{device_name}_dataframe.sav")
        length_temp_shuffled = len(temp_shuffled.index)
        for current_fold in range(5):
            x[device_name][current_fold] = []
            y[device_name][current_fold] = []
            if current_fold == 0:
                # Get 20% for use in testing
                temp_shuffled_test = temp_shuffled[:int(length_temp_shuffled * .2)]
            elif current_fold == 1:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .2):int(length_temp_shuffled * .4)]
            elif current_fold == 2:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .4):int(length_temp_shuffled * .6)]
            elif current_fold == 3:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .6):int(length_temp_shuffled * .8)]
            else:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .8):]

            x[device_name][current_fold] = x[device_name][current_fold] + temp_shuffled_test.drop(['class'], axis=1).values.tolist()
            y[device_name][current_fold] = y[device_name][current_fold] + temp_shuffled_test['class'].values.tolist()

    accuracy_dict = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    model_path = f"./saved_data/models/{name_of_current_data}/{dataset_dict[dataset_name]}"
    model_list = os.listdir(model_path)
    for model_index, model_name in enumerate(sorted(model_list)):
        model = pickle.load(open(model_path+"/"+model_name, 'rb'))
        # result = loaded_model.score(x[model_name]["test"], y[model_name]["test"])
        # print(result)

        print(f"*** Begin Fold {model_index} ***")

        fold_accuracy = 0

        for device_name in labels_numeric:
            if f"device {class_label_dict[device_name]}" not in accuracy_dict:
                accuracy_dict[f"device {class_label_dict[device_name]}"] = 0
            if f"device {class_label_dict[device_name]}" not in precision_dict:
                precision_dict[f"device {class_label_dict[device_name]}"] = 0
            if f"device {class_label_dict[device_name]}" not in recall_dict:
                recall_dict[f"device {class_label_dict[device_name]}"] = 0
            if f"device {class_label_dict[device_name]}" not in f1_dict:
                f1_dict[f"device {class_label_dict[device_name]}"] = 0

            y_pred = model.predict(x[device_name][model_index])

            accuracy_dict[f"device {class_label_dict[device_name]}"] += accuracy_score(y[device_name][model_index], y_pred)

            fold_accuracy += accuracy_score(y[device_name][model_index], y_pred)

            precision_dict[f"device {class_label_dict[device_name]}"] += precision_score(y[device_name][model_index], y_pred, average='weighted')
            recall_dict[f"device {class_label_dict[device_name]}"] += recall_score(y[device_name][model_index], y_pred, average='weighted')
            f1_dict[f"device {class_label_dict[device_name]}"] += f1_score(y[device_name][model_index], y_pred, average='weighted')

        print("*****")
        print()
        print(f"Fold Accuracy: {fold_accuracy / 22}")
        print()
        print("*****")

        print(f"*** Completed Fold {model_index} ***")

    total_accuracy = 0
    for device_name in labels_numeric:
        accuracy_dict[f"device {class_label_dict[device_name]}"] /= len(model_list)

        total_accuracy+=accuracy_dict[f"device {class_label_dict[device_name]}"]

        precision_dict[f"device {class_label_dict[device_name]}"] /= len(model_list)
        recall_dict[f"device {class_label_dict[device_name]}"] /= len(model_list)
        f1_dict[f"device {class_label_dict[device_name]}"] /= len(model_list)

        wandb.log({f"{class_label_dict[device_name]} Total accuracy on {name_of_current_data}": accuracy_dict[f"device {class_label_dict[device_name]}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"{class_label_dict[device_name]} Total precision on {name_of_current_data}": precision_dict[f"device {class_label_dict[device_name]}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"{class_label_dict[device_name]} Total recall on {name_of_current_data}": recall_dict[f"device {class_label_dict[device_name]}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"{class_label_dict[device_name]} Total f1 on {name_of_current_data}": f1_dict[f"device {class_label_dict[device_name]}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
    # print("*****")
    # print()
    # print(f"Total Accuracy: {total_accuracy/22}")
    # print()
    # print("*****")