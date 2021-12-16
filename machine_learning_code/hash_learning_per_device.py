from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import wandb
import os

wandb.init(project="smart_attacker", entity="unr-mpl")

# Get hash csv file paths
path_to_csvs = "../hashes_cleaned/"
# path_to_csvs = "../hashes_uncleaned/"
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
    labels = dataset['class'].unique()
    print("*** Dataset Loaded ***")

    # # Split the dataset into 70 percent train 30 percent test
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x.values.tolist(),
    #     y.values.tolist(),
    #     test_size=0.0)

    x_train = x.values.tolist()
    y_train = y.values.tolist()

    # Spot Check Algorithms
    models = []
    models.append(('MLP', MLPClassifier()))

    model_dict = {}
    model_count = 0
    # evaluate each model
    for model_name, model in models:
        model_dict[model_name] = model_count
        model_count += 1
        print(f"*** Begin Training {model_name} ***")
        model.fit(x_train, y_train)
        print(f"*** {model_name} Trained ***")

        for device_name in labels:
            # making boolean series for a team name
            filter = dataset["class"] == device_name
            # filtering data
            device_dataset = dataset
            device_dataset = dataset.where(filter)
            device_dataset.dropna(inplace=True)

            x_device = device_dataset.drop(['class'], axis=1)
            y_device = device_dataset["class"]

            # # Split the dataset into 70 percent train 30 percent test
            # x_train_device, x_test_device, y_train_device, y_test_device = train_test_split(
            #     x_device.values.tolist(),
            #     y_device.values.tolist(),
            #     test_size=1.0)

            x_test_device = x_device.values.tolist()
            y_test_device = y_device.values.tolist()

            print(f"*** Calculate Predictions and Probabilities ***")
            y_pred_device = model.predict(x_test_device)
            y_probas_device = model.predict_proba(x_test_device)
            print(f"*** Predictions and Probabilities Done ***")

            print("*** Begin Metric Plotting ***")

            accuracy = accuracy_score(y_test_device, y_pred_device)
            precision = precision_score(y_test_device, y_pred_device, average='weighted')
            recall = recall_score(y_test_device, y_pred_device, average='weighted')
            f1 = f1_score(y_test_device, y_pred_device, average='weighted')

            wandb.log({f"{model_name} Accuracy on Cleaned Hashes for device {device_name}": accuracy, "Dataset": dataset_dict[dataset_name], "Num Samples": dataset.shape[0]})
            wandb.log({f"{model_name} Precision Weighted on Cleaned Hashes for device {device_name}": precision, "Dataset": dataset_dict[dataset_name], "Num Samples": dataset.shape[0]})
            wandb.log({f"{model_name} Recall Weighted on Cleaned Hashes for device {device_name}": recall, "Dataset": dataset_dict[dataset_name], "Num Samples": dataset.shape[0]})
            wandb.log({f"{model_name} F1 Weighted on Cleaned Hashes for device {device_name}": f1, "Dataset": dataset_dict[dataset_name], "Num Samples": dataset.shape[0]})

            print("*** Metric Plotting Completed ***")
