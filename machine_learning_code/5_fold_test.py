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
# name_of_current_data = "Cleaned Hashes No Ring"
# name_of_current_data = "Cleaned Hashes No Ring No TpBulb"
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

    for unique_value in sorted(y.unique()):
        for index, value in enumerate(y):
            if value == unique_value:
                y_temp[index] = counter
        counter += 1

    dataset["class"] = y_temp
    y = dataset['class']
    labels_numeric = dataset['class'].unique()

    print("*** Dataset Loaded ***")

    print("*** Begin Generating Cross Folds ***")
    if not os.path.isdir(f"./saved_data/dataframes/{name_of_current_data}"):
        os.mkdir(f"./saved_data/dataframes/{name_of_current_data}")
    if not os.path.isdir(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}"):
        os.mkdir(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}")
    # Create datasets for 5 fold cross validation
    # x_devices = {}
    # y_devices = {}

    x = {0:{"train":[], "test":[]}, 1:{"train":[], "test":[]}, 2:{"train":[], "test":[]},
         3:{"train":[], "test":[]}, 4:{"train":[], "test":[]}, 5:{"train":[], "test":[]}}
    y = {0:{"train":[], "test":[]}, 1:{"train":[], "test":[]}, 2:{"train":[], "test":[]},
         3:{"train":[], "test":[]}, 4:{"train":[], "test":[]}, 5:{"train":[], "test":[]}}

    x_device = {0:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    y_device = {0:{}, 2:{}, 3:{}, 4:{}, 5:{}}

    for device_name in labels_numeric:
        temp_shuffled = pd.read_pickle(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}/"
                                       f"shuffled_{name_of_current_data}-{dataset_dict[dataset_name]}_device-{device_name}_dataframe.sav")
        length_temp_shuffled = len(temp_shuffled.index)
        for current_fold in range(5):
            if current_fold == 0:
                # Get 20% for use in testing
                temp_shuffled_test = temp_shuffled[:int(length_temp_shuffled * .2)]
                # Get 80% for use in training
                temp_shuffled_train = temp_shuffled[int(length_temp_shuffled * .2):]
            elif current_fold == 1:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .2):int(length_temp_shuffled * .4)]
                dataframes = [temp_shuffled[:int(length_temp_shuffled * .2)], temp_shuffled[int(length_temp_shuffled * .4):]]
                temp_shuffled_train = pd.concat(dataframes, ignore_index=True)
            elif current_fold == 2:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .4):int(length_temp_shuffled * .6)]
                dataframes = [temp_shuffled[:int(length_temp_shuffled * .2)], temp_shuffled[int(length_temp_shuffled * .4):]]
                temp_shuffled_train = pd.concat(dataframes, ignore_index=True)
            elif current_fold == 3:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .6):int(length_temp_shuffled * .8)]
                dataframes = [temp_shuffled[:int(length_temp_shuffled * .6)], temp_shuffled[int(length_temp_shuffled * .8):]]
                temp_shuffled_train = pd.concat(dataframes, ignore_index=True)
            else:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .8):]
                temp_shuffled_train = temp_shuffled[:int(length_temp_shuffled * .8)]

            x[current_fold]["test"] = (x[current_fold]["test"] + temp_shuffled_test.drop(['class'], axis=1).values.tolist())
            y[current_fold]["test"] = (y[current_fold]["test"] + temp_shuffled_test['class'].values.tolist())
            x[current_fold]["train"] = (x[current_fold]["train"] + temp_shuffled_train.drop(['class'], axis=1).values.tolist())
            y[current_fold]["train"] = (y[current_fold]["train"] + temp_shuffled_train['class'].values.tolist())

            # Randomly shuffle the resulting training dataset so that all samples for the same class are not passed in together

            temp2 = list(zip(x[current_fold]["train"], y[current_fold]["train"]))
            random.shuffle(temp2)
            x[current_fold]["train"], y[current_fold]["train"] = [[ i for i, j in temp2], [ j for i, j in temp2]]
            # x[current_fold]["train"], y[current_fold]["train"] = zip(*temp)
    print("*** Finished Generating Cross Folds ***")

    accuracy_dict = {"base": 0}
    precision_dict = {"base": 0}
    recall_dict = {"base": 0}
    f1_dict = {"base": 0}

    model_path = f"./saved_data/models/{name_of_current_data}/{dataset_dict[dataset_name]}"
    model_list = os.listdir(model_path)
    # evaluate each model
    for model_index, model_name in enumerate(sorted(model_list)):
        model = pickle.load(open(model_path + "/" + model_name, 'rb'))

        # print(f"*** Calculate Predictions and Probabilities ***")
        y_pred = model.predict(x[model_index]["test"])
        y_probas = model.predict_proba(x[model_index]["test"])
        # print(f"*** Predictions and Probabilities Done ***")

        accuracy_dict[f"fold {model_index}"] = accuracy_score(y[model_index]["test"], y_pred)
        accuracy_dict["base"] += accuracy_score(y[model_index]["test"], y_pred)
        precision_dict[f"fold {model_index}"] = precision_score(y[model_index]["test"], y_pred, average='weighted')
        precision_dict[f"base"] += precision_score(y[model_index]["test"], y_pred, average='weighted')
        recall_dict[f"fold {model_index}"] = recall_score(y[model_index]["test"], y_pred, average='weighted')
        recall_dict["base"] += recall_score(y[model_index]["test"], y_pred, average='weighted')
        f1_dict[f"fold {model_index}"] = f1_score(y[model_index]["test"], y_pred, average='weighted')
        f1_dict["base"] += f1_score(y[model_index]["test"], y_pred, average='weighted')


        print(f"Fold {model_index} accuracy on  {name_of_current_data}: {accuracy_dict[f'fold {model_index}']}")

    # print("*** Begin Metric Plotting ***")

    accuracy_dict["base"] /= len(model_list)

    print(f"Total accuracy on {name_of_current_data}: {accuracy_dict['base']}")
