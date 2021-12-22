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
path_to_csvs = "../hashes_cleaned_confusionMatrix/"
# path_to_csvs = "../hashes_cleaned_noRing/"
# path_to_csvs = "../hashes_cleaned_noRing_noTpBulb/"
# path_to_csvs = "../hashes_uncleaned/"

csv_names = sorted(os.listdir(path_to_csvs), reverse=False)[1:]
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

    x_devices = {}
    y_devices = {}

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for device_name in labels_numeric:
        temp = dataset[dataset['class'] == device_name]
        temp_shuffled = temp.sample(frac=1)
        temp_shuffled_head = temp_shuffled.head(int((len(temp_shuffled.index))*.3))
        temp_shuffled_tail = temp_shuffled.tail(int((len(temp_shuffled.index)) * .7))

        x_devices[device_name] = temp_shuffled_head.drop(['class'], axis=1).values.tolist()
        y_devices[device_name] = temp_shuffled_head['class'].values.tolist()

        x_train = x_train + temp_shuffled_tail.drop(['class'], axis=1).values.tolist()
        y_train  = y_train + temp_shuffled_tail['class'].values.tolist()
        x_test = x_test + temp_shuffled_tail.drop(['class'], axis=1).values.tolist()
        y_test = y_test + temp_shuffled_tail['class'].values.tolist()

    temp = list(zip(x_train, y_train))
    random.shuffle(temp)
    x_train, y_train = zip(*temp)
    temp = list(zip(x_test, y_test))
    random.shuffle(temp)
    x_test, y_test = zip(*temp)

    num_samples_dev = []
    for device_name in sorted(labels_numeric):
        num_samples_dev.append(len(temp_shuffled.index))
        temp = dataset[dataset['class'] == device_name]
        temp_shuffled = temp.sample(frac=1)
        temp_shuffled_head = temp_shuffled.head(int((len(temp_shuffled.index)) * .3))
        x_devices[device_name] = temp_shuffled_head.drop(['class'], axis=1).values.tolist()
        y_devices[device_name] = temp_shuffled_head['class'].values.tolist()

    # Spot Check Algorithms
    models = []
    models.append(('MLP', MLPClassifier()))
    temp = MLPClassifier()
    print(temp.get_params())

    accuracy_dict = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    model_dict = {}
    model_count = 0
    # evaluate each model
    for model_name, model in models:
        model_dict[model_name] = model_count
        model_count += 1
        print(f"*** Begin Training {model_name} ***")
        model.fit(x_train, y_train)
        print(f"*** {model_name} Trained ***")

        print(f"*** Calculate Predictions and Probabilities ***")
        y_pred = model.predict(x_test)
        y_probas = model.predict_proba(x_test)
        print(f"*** Predictions and Probabilities Done ***")

        plt.autoscale()
        fig, ax = plt.subplots(figsize=(20, 20))

        conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
        np.save("cleaned_oneMinute_confusion_matrix_devicesRemoved", conf_mat)
        ax = sns.heatmap(conf_mat, cbar=False, annot=True, xticklabels=labels_string, yticklabels=labels_string)

        # fig.savefig(dataset_name[20:-4] + "_uncleaned.png")
        # fig.savefig(dataset_name[34:-4] + "_noRing.png")

        exit(0)

        # print(f"*** Calculate 5 Fold Accuracy and F1 ***")
        # kfold = StratifiedKFold(n_splits=5, shuffle=True)
        # cv_accuracy = cross_val_score(model, x.values.tolist(), y.values.tolist(), cv=kfold, scoring='accuracy',
        #                               n_jobs=-1)
        # cv_f1 = cross_val_score(model, x.values.tolist(), y.values.tolist(), cv=kfold, scoring='f1_weighted', n_jobs=-1)
        #
        # cv_accuracy_average = sum(cv_accuracy) / len(cv_accuracy)
        # cv_f1_average = sum(cv_f1) / len(cv_f1)
        # print(f"*** 5 Fold Accuracy and F1 Done ***")

        accuracy_dict["base"] = accuracy_score(y_test, y_pred)
        precision_dict["base"] = precision_score(y_test, y_pred, average='weighted')
        recall_dict["base"] = recall_score(y_test, y_pred, average='weighted')
        f1_dict["base"] = f1_score(y_test, y_pred, average='weighted')

        print("*** Begin Per Device ***")
        for device_name in sorted(labels_numeric):
            x_test_device = x_devices[device_name]
            y_test_device = y_devices[device_name]

            y_pred_device = model.predict(x_test_device)
            y_probas_device = model.predict_proba(x_test_device)

            accuracy_dict[device_name] = accuracy_score(y_test_device, y_pred_device)
            precision_dict[device_name] = precision_score(y_test_device, y_pred_device, average='weighted')
            recall_dict[device_name] = recall_score(y_test_device, y_pred_device, average='weighted')
            f1_dict[device_name] = f1_score(y_test_device, y_pred_device, average='weighted')
        print("*** Per Device Completed ***")

        print("*** Begin Metric Plotting ***")

        wandb.log({f"{model_name} accuracy on Uncleaned Hashes": accuracy_dict["base"],
                   f"{class_label_dict[0]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[0],
                   f"{class_label_dict[1]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[1],
                   f"{class_label_dict[2]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[2],
                   f"{class_label_dict[3]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[3],
                   f"{class_label_dict[4]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[4],
                   f"{class_label_dict[5]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[5],
                   f"{class_label_dict[6]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[6],
                   f"{class_label_dict[7]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[7],
                   f"{class_label_dict[8]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[8],
                   f"{class_label_dict[9]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[9],
                   f"{class_label_dict[10]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[10],
                   f"{class_label_dict[11]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[11],
                   f"{class_label_dict[12]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[12],
                   f"{class_label_dict[13]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[13],
                   f"{class_label_dict[14]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[14],
                   f"{class_label_dict[15]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[15],
                   f"{class_label_dict[16]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[16],
                   f"{class_label_dict[17]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[17],
                   f"{class_label_dict[18]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[18],
                   f"{class_label_dict[19]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[19],
                   f"{class_label_dict[20]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[20],
                   f"{class_label_dict[21]} {model_name} accuracy on Uncleaned Hashes": accuracy_dict[21],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"{model_name} precision on Uncleaned Hashes": precision_dict["base"],
                   f"{class_label_dict[0]} {model_name} precision on Uncleaned Hashes": precision_dict[0],
                   f"{class_label_dict[1]} {model_name} precision on Uncleaned Hashes": precision_dict[1],
                   f"{class_label_dict[2]} {model_name} precision on Uncleaned Hashes": precision_dict[2],
                   f"{class_label_dict[3]} {model_name} precision on Uncleaned Hashes": precision_dict[3],
                   f"{class_label_dict[4]} {model_name} precision on Uncleaned Hashes": precision_dict[4],
                   f"{class_label_dict[5]} {model_name} precision on Uncleaned Hashes": precision_dict[5],
                   f"{class_label_dict[6]} {model_name} precision on Uncleaned Hashes": precision_dict[6],
                   f"{class_label_dict[7]} {model_name} precision on Uncleaned Hashes": precision_dict[7],
                   f"{class_label_dict[8]} {model_name} precision on Uncleaned Hashes": precision_dict[8],
                   f"{class_label_dict[9]} {model_name} precision on Uncleaned Hashes": precision_dict[9],
                   f"{class_label_dict[10]} {model_name} precision on Uncleaned Hashes": precision_dict[10],
                   f"{class_label_dict[11]} {model_name} precision on Uncleaned Hashes": precision_dict[11],
                   f"{class_label_dict[12]} {model_name} precision on Uncleaned Hashes": precision_dict[12],
                   f"{class_label_dict[13]} {model_name} precision on Uncleaned Hashes": precision_dict[13],
                   f"{class_label_dict[14]} {model_name} precision on Uncleaned Hashes": precision_dict[14],
                   f"{class_label_dict[15]} {model_name} precision on Uncleaned Hashes": precision_dict[15],
                   f"{class_label_dict[16]} {model_name} precision on Uncleaned Hashes": precision_dict[16],
                   f"{class_label_dict[17]} {model_name} precision on Uncleaned Hashes": precision_dict[17],
                   f"{class_label_dict[18]} {model_name} precision on Uncleaned Hashes": precision_dict[18],
                   f"{class_label_dict[19]} {model_name} precision on Uncleaned Hashes": precision_dict[19],
                   f"{class_label_dict[20]} {model_name} precision on Uncleaned Hashes": precision_dict[20],
                   f"{class_label_dict[21]} {model_name} precision on Uncleaned Hashes": precision_dict[21],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"{model_name} recall on Uncleaned Hashes": recall_dict["base"],
                   f"{class_label_dict[0]} {model_name} recall on Uncleaned Hashes": recall_dict[0],
                   f"{class_label_dict[1]} {model_name} recall on Uncleaned Hashes": recall_dict[1],
                   f"{class_label_dict[2]} {model_name} recall on Uncleaned Hashes": recall_dict[2],
                   f"{class_label_dict[3]} {model_name} recall on Uncleaned Hashes": recall_dict[3],
                   f"{class_label_dict[4]} {model_name} recall on Uncleaned Hashes": recall_dict[4],
                   f"{class_label_dict[5]} {model_name} recall on Uncleaned Hashes": recall_dict[5],
                   f"{class_label_dict[6]} {model_name} recall on Uncleaned Hashes": recall_dict[6],
                   f"{class_label_dict[7]} {model_name} recall on Uncleaned Hashes": recall_dict[7],
                   f"{class_label_dict[8]} {model_name} recall on Uncleaned Hashes": recall_dict[8],
                   f"{class_label_dict[9]} {model_name} recall on Uncleaned Hashes": recall_dict[9],
                   f"{class_label_dict[10]} {model_name} recall on Uncleaned Hashes": recall_dict[10],
                   f"{class_label_dict[11]} {model_name} recall on Uncleaned Hashes": recall_dict[11],
                   f"{class_label_dict[12]} {model_name} recall on Uncleaned Hashes": recall_dict[12],
                   f"{class_label_dict[13]} {model_name} recall on Uncleaned Hashes": recall_dict[13],
                   f"{class_label_dict[14]} {model_name} recall on Uncleaned Hashes": recall_dict[14],
                   f"{class_label_dict[15]} {model_name} recall on Uncleaned Hashes": recall_dict[15],
                   f"{class_label_dict[16]} {model_name} recall on Uncleaned Hashes": recall_dict[16],
                   f"{class_label_dict[17]} {model_name} recall on Uncleaned Hashes": recall_dict[17],
                   f"{class_label_dict[18]} {model_name} recall on Uncleaned Hashes": recall_dict[18],
                   f"{class_label_dict[19]} {model_name} recall on Uncleaned Hashes": recall_dict[19],
                   f"{class_label_dict[20]} {model_name} recall on Uncleaned Hashes": recall_dict[20],
                   f"{class_label_dict[21]} {model_name} recall on Uncleaned Hashes": recall_dict[21],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"{model_name} f1 on Uncleaned Hashes": f1_dict["base"],
                   f"{class_label_dict[0]} {model_name} f1 on Uncleaned Hashes": f1_dict[0],
                   f"{class_label_dict[1]} {model_name} f1 on Uncleaned Hashes": f1_dict[1],
                   f"{class_label_dict[2]} {model_name} f1 on Uncleaned Hashes": f1_dict[2],
                   f"{class_label_dict[3]} {model_name} f1 on Uncleaned Hashes": f1_dict[3],
                   f"{class_label_dict[4]} {model_name} f1 on Uncleaned Hashes": f1_dict[4],
                   f"{class_label_dict[5]} {model_name} f1 on Uncleaned Hashes": f1_dict[5],
                   f"{class_label_dict[6]} {model_name} f1 on Uncleaned Hashes": f1_dict[6],
                   f"{class_label_dict[7]} {model_name} f1 on Uncleaned Hashes": f1_dict[7],
                   f"{class_label_dict[8]} {model_name} f1 on Uncleaned Hashes": f1_dict[8],
                   f"{class_label_dict[9]} {model_name} f1 on Uncleaned Hashes": f1_dict[9],
                   f"{class_label_dict[10]} {model_name} f1 on Uncleaned Hashes": f1_dict[10],
                   f"{class_label_dict[11]} {model_name} f1 on Uncleaned Hashes": f1_dict[11],
                   f"{class_label_dict[12]} {model_name} f1 on Uncleaned Hashes": f1_dict[12],
                   f"{class_label_dict[13]} {model_name} f1 on Uncleaned Hashes": f1_dict[13],
                   f"{class_label_dict[14]} {model_name} f1 on Uncleaned Hashes": f1_dict[14],
                   f"{class_label_dict[15]} {model_name} f1 on Uncleaned Hashes": f1_dict[15],
                   f"{class_label_dict[16]} {model_name} f1 on Uncleaned Hashes": f1_dict[16],
                   f"{class_label_dict[17]} {model_name} f1 on Uncleaned Hashes": f1_dict[17],
                   f"{class_label_dict[18]} {model_name} f1 on Uncleaned Hashes": f1_dict[18],
                   f"{class_label_dict[19]} {model_name} f1 on Uncleaned Hashes": f1_dict[19],
                   f"{class_label_dict[20]} {model_name} f1 on Uncleaned Hashes": f1_dict[20],
                   f"{class_label_dict[21]} {model_name} f1 on Uncleaned Hashes": f1_dict[21],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})

        print("*** Metric Plotting Completed ***")
