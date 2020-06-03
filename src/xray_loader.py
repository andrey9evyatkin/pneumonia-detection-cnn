from pathlib import Path

from constants import *


def load_data():
    path = Path(DATA_PATH)
    train_data = [load_data_cases(path / TRAIN_DATA_PATH), 'train']
    val_data = [load_data_cases(path / VAL_DATA_PATH), 'val']
    test_data = [load_data_cases(path / TEST_DATA_PATH), 'test']
    return train_data, val_data, test_data


def load_data_cases(dir_path):
    normal_cases_dir = dir_path / DATA_NORMAL_CASES_PATH
    pneumonia_cases_dir = dir_path / DATA_PNEUMONIA_CASES_PATH
    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
    return normal_cases, pneumonia_cases

