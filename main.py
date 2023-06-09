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

    def get_entropy(self, neg_examples, pos_examples):
        ttl_examples = neg_examples + pos_examples
        p_neg = neg_examples / ttl_examples
        p_pos = pos_examples / ttl_examples
        return - (p_neg * np.log2(p_neg)) - (p_pos * np.log2(p_pos))

    def get_entropies_for_feature(self, feature):
        entropys_per_option = []
        for name, value in self.count_dict[feature].items():
            predicted_neg = value[0]  # negative predicted value when option == name
            predicted_pos = value[1]  # positive predicted value when option == name
            entropys_per_option.append({name: self.get_entropy(neg_examples=predicted_neg, pos_examples=predicted_pos)})
        return entropys_per_option


    def get_gain(self):
        pass
#TODO: check if we can assume yes or no in predicted feature

    def count_features(self, examples):
        feature_options = {}
        for example in examples:
            for feature, option in example.items():
                if feature not in feature_options:
                    feature_options[feature] = {}
                if option not in feature_options[feature]:
                    feature_options[feature][option] = [0, 0]
                if example[self.feature_to_predict] == 'no' or example[self.feature_to_predict] == '0':
                    feature_options[feature][option][0] += 1
                else:
                    feature_options[feature][option][1] += 1
        return feature_options


if __name__ == '__main__':
    file_path = 'train.txt'
    p = Prediction(file_path)
    entropies_age = p.get_entropies_for_feature('age')

    p = 0
