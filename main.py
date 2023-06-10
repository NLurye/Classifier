import pandas as pd
import numpy as np


class Prediction:

    def __init__(self, train_data):
        self.info_gains = None
        self.all_entropies = None
        self.feature_to_predict, self.ttl_examples, self.train_examples = self.parse_data(train_data)
        self.count_dict = self.count_features(self.train_examples)



    class Tree:
        def __init__(self):
            pass

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
        if neg_examples == 0 or pos_examples == 0:
            return 0.0  # Return 0 entropy for zero examples
        p_neg = neg_examples / ttl_examples
        p_pos = pos_examples / ttl_examples
        return - (p_neg * np.log2(p_neg)) - (p_pos * np.log2(p_pos))

    def get_entropies_for_feature(self, feature):
        feature_entropys = {}
        ttl_predicted_neg = 0
        ttl_predicted_pos = 0
        for name, value in self.count_dict[feature].items():
            predicted_neg = value[0]  # negative predicted value when option == name
            predicted_pos = value[1]  # positive predicted value when option == name
            ttl_predicted_neg += predicted_neg
            ttl_predicted_pos += predicted_pos
            feature_entropys.setdefault('entropies_per_option', {})[name] = {
                'entropy': self.get_entropy(neg_examples=predicted_neg, pos_examples=predicted_pos),
                'ttl_count_for_option': predicted_neg + predicted_pos
            }
            feature_entropys['entropies_per_option'][name]['pure'] = \
                True if feature_entropys['entropies_per_option'][name]['entropy'] == 0 else False
        feature_entropys['ttl_entropy_for_feature'] = self.get_entropy(neg_examples=ttl_predicted_neg,
                                                                       pos_examples=ttl_predicted_pos)
        return feature_entropys

    def get_gain_for_feature(self, entropies_for_feature):
        ttl_feature_gain = entropies_for_feature['ttl_entropy_for_feature']
        for option in entropies_for_feature['entropies_per_option'].values():
            ttl_feature_gain -= (option['ttl_count_for_option'] / self.ttl_examples) * option['entropy']
        return ttl_feature_gain

    def calc_entropies_for_all(self):
        self.all_entropies = {}
        for feature in self.count_dict.keys():
            self.all_entropies[feature] = self.get_entropies_for_feature(feature)

    def calc_gain_for_all(self):
        self.calc_entropies_for_all()
        info_gains = {}
        for feature, entropies in self.all_entropies.items():
            info_gains[feature] = self.get_gain_for_feature(entropies_for_feature=entropies)
        self.info_gains = dict(sorted(info_gains.items(), key=lambda item: item[1], reverse=True))
    # TODO: check if we can assume yes or no in predicted feature

    def count_features(self, examples):
        feature_options = {}
        for example in examples:
            for feature, option in example.items():
                if feature != self.feature_to_predict:
                    if feature not in feature_options:
                        feature_options[feature] = {}
                    if option not in feature_options[feature]:
                        feature_options[feature][option] = [0, 0]
                    if example[self.feature_to_predict] == 'no' or example[self.feature_to_predict] == '0':
                        feature_options[feature][option][0] += 1
                    else:
                        feature_options[feature][option][1] += 1
        return feature_options


# TODO: Temp. gain 0.06423863677845937

if __name__ == '__main__':
    file_path = 'test2.txt'
    p = Prediction(file_path)
    p.calc_gain_for_all()
    p = 0
