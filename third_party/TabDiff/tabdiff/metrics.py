"""Training-table metadata for MP-TabDiff sampling."""

import pandas as pd


class TabMetrics:
    def __init__(self, real_data_path, test_data_path, val_data_path, info, device, metric_list):
        self.info = info
        self.real_data_size = len(pd.read_csv(real_data_path))

    def evaluate(self, syn_data):
        return {}, {}
