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
import random
from tqdm import tqdm

columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11',
           'dim12', 'dim13', 'dim14', 'dim15', 'dim16', 'dim17', 'dim18', 'dim19', 'dim20', 'dim21',
           'dim22', 'dim23', 'dim24', 'dim25', 'dim26', 'dim27', 'dim28', 'dim29', 'dim30', 'dim31', 'dim32', 'class']

path_to_csv = "../hashes_cleaned/0-minute-hashes-cleaned.csv"

dataset = read_csv(path_to_csv)
data_length = dataset.shape[0]

hash_noise_list = []
for index in tqdm(range(data_length*10)):
    hash_noise_list.append(random.sample(range(0, 256), 32))
    hash_noise_list[index].append(22)

hash_noise_df = pd.DataFrame(hash_noise_list)

hash_noise_df.to_csv(f"hash_noise.csv")

