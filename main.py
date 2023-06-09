import pandas as pd
import numpy as np


class Prediction:

    def __init__(self, train_data):
        self.feature_to_predict, self.ttl_examples, self.train_examples = self.parse_data(train_data)
        self.count_dict = self.count_features(self.train_examples)

    def parse_data(self, txt):
        with open(txt, 'r') as file:
            lines = file.readlines()
        data = []
        header = lines[0].replace("\n", "").split('\t')
        for line in lines[1:]:
            values = line.replace("\n", "").split('\t')
            data.append(dict(zip(header, values)))
        return header[-1], len(data), data

    def get_entropy(self, pos_examples, neg_examples):
        p_neg = neg_examples / self.ttl_examples
        p_pos = pos_examples / self.ttl_examples
        return - (p_neg * np.log2(p_neg)) - (p_pos * np.log2(p_pos))

    def get_ttl_entropy(self):
        self.get_entropy()

    def get_gain(self):
        pass

    def count_features(self, examples, feature_to_predict):
        feature_options = {}
        for example in examples:
            for feature, option in example.items():
                if feature not in feature_options:
                    feature_options[feature] = {}
                if option not in feature_options[feature]:
                    feature_options[feature][option] = (0, 0)
                if example[feature_to_predict] == None:
                    feature_options[feature][option] += 1
        return feature_options


if __name__ == '__main__':
    file_path = 'train.txt'
    p = Prediction(file_path)
    p = 0
