"""util function for the project"""
import pickle

from pandas import DataFrame


def load_binary_data(file_name: str):
    """load binary data"""
    with open(file_name, 'rb') as file_read:
        data = pickle.load(file_read)
        print(type(data))
        return data


def expand_label_cols(labels: DataFrame):
    """expand labels row by serious and morethan3"""
    labels['serious'] = labels['count'] > 1
    labels['serious'].astype(int)
    labels['morethan3'] = labels['count'] > 2
    labels['morethan3'].astype(int)
    print(labels['serious'].value_counts())
    print(labels['morethan3'].value_counts())
