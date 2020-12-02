import pandas as pd
import numpy as np
import os
import argparse
import _utils
from sklearn.model_selection import train_test_split

"""
Splits input data into n folds for Flair model training and later ML.
Example use:

python3 input_data_splitter.py --folds=5 --dependent_variable=respect
"""

parser = argparse.ArgumentParser(description='split_input_data')
parser.add_argument("--dependent_variable", required=False, default="leadership", type=str)
parser.add_argument("--folds", required=False, default=5, type=int)
parser.add_argument("--filepath", required=False, default='5k_final_leadership_values.xlsx', type=str)
parser.add_argument("--binarize", required=False, default=False, action='store_true')
args = parser.parse_args()

filepath = os.path.join('./data/source_data/', args.filepath)
data_file_name = args.filepath.split('.')[0]
data_file_extension = args.filepath.split('.')[1]
dependent_variable = args.dependent_variable
folds = args.folds
binarize = args.binarize

# read original data file
if data_file_extension == "xlsx":
    df = pd.read_excel(filepath, converters={'id': str})
elif data_file_extension == "csv":
    df = pd.read_csv(filepath, converters={'id': str})

# use only selected columns
df = df[[dependent_variable, "text"]]

if binarize:
    df[dependent_variable] = df[dependent_variable].map(lambda x: _utils.binarize(x))

# change format of 'sentiment' label for further training in Flair framework
df[dependent_variable] = '__label__' + df[dependent_variable].astype(str)

# Cross validation
# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = dict()
val_indexes = dict()
test_indexes = dict()

# train sets for Machine Learning
train_ml = dict()

kf = _utils.skf_splitter(folds=folds, df=df, dependent_variable=dependent_variable)

i = 0
for train_index, test_index in kf:
    test_indexes[i] = test_index
    train_ml[i] = train_index
    train_index, val_index = train_test_split(train_index, test_size=0.125, random_state=13, shuffle=True,
                                              stratify=df.iloc[train_index][dependent_variable])
    train_indexes[i] = train_index
    val_indexes[i] = val_index
    i += 1
    
# test sets for Machine Learning are equal to those for Flair framework
test_ml = test_indexes

# create folders for FLAIR data splits and .tsv files for training
folds_path = list()
for fold in range(folds):
    folds_path.append(f'./data/models/{dependent_variable}_{data_file_name}_{str(fold)}/')
    try:
        os.mkdir(f'./data/models/{dependent_variable}_{data_file_name}_{str(fold)}')
    except FileExistsError:
        pass  # continue
    df.iloc[test_indexes[fold]].to_csv(os.path.join(folds_path[fold], "test_.tsv"),
                                       index=False, header=False, encoding='utf-8', sep='\t')
    df.iloc[train_indexes[fold]].to_csv(os.path.join(folds_path[fold], "train.tsv"),
                                        index=False, header=False, encoding='utf-8', sep='\t')
    df.iloc[val_indexes[fold]].to_csv(os.path.join(folds_path[fold], "dev.tsv"),
                                      index=False, header=False, encoding='utf-8', sep='\t')
