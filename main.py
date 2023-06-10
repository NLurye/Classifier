import pandas as pd
import numpy as np


class Tree:
    def __init__(self, attribute=None, value=None):
        self.attribute = attribute  # Feature associated with the node
        self.value = value  # Value of the predicted feature (for leaf nodes)
        self.children = {}  # Dictionary to store child nodes

    def add_child(self, value, subtree):
        self.children[value] = subtree

    def predict(self, example):
        if self.value is not None:
            return self.value  # Reached a leaf node, return the predicted value

        attribute_value = example[self.attribute]
        if attribute_value not in self.children:
            return None  # Unknown attribute value, return None

        child_subtree = self.children[attribute_value]
        return child_subtree.predict(example)

    def print_tree(self, indent=''):
        if self.value is not None:
            print(indent + str(self.value))
            return

        print(indent + str(self.attribute) + ' {')
        for value, subtree in self.children.items():
            print(indent + '  ' + str(value) + ':', end=' ')
            subtree.print_tree(indent + '    ')
        print(indent + '}')


class Prediction:

    def __init__(self, train_data):
        self.info_gains = None
        self.all_entropies = None
        self.feature_to_predict, self.ttl_examples, self.train_examples = self.parse_data(train_data)
        self.count_dict = self.count_features(self.train_examples)
        self.tree = Tree()

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

    def get_entropies_for_feature(self, feature, examples):
        feature_entropys = {}
        ttl_predicted_neg = 0
        ttl_predicted_pos = 0
        count_sub_dict = self.count_features(examples)
        for name, value in count_sub_dict[feature].items():
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

    def calc_entropies_for_all(self, features, examples):
        all_entropies = {}
        for feature in features:
            all_entropies[feature] = self.get_entropies_for_feature(feature, examples)
        return all_entropies

    def calc_gain_for_all(self, features, examples):
        self.calc_entropies_for_all(features, examples)
        info_gains = {}
        for feature, entropies in self.all_entropies.items():
            info_gains[feature] = self.get_gain_for_feature(entropies_for_feature=entropies)
        return info_gains
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

    def get_most_informative(self, gains):
        return max(gains, key=gains.get)

    def make_tree(self):
        features = list(self.count_dict.keys())
        self.all_entropies = self.calc_entropies_for_all(features, self.train_examples)
        self.info_gains = self.calc_gain_for_all(features, self.train_examples)
        self.tree = self.build_tree(self.train_examples, self.info_gains)

    def build_tree(self, examples, info_gains):
        feature_to_predict = self.feature_to_predict
        if len(set([example[feature_to_predict] for example in examples])) == 1:
            # All examples have the same value for the predicted feature
            return Tree(value=examples[0][feature_to_predict])

        if len(examples[0]) == 1:
            # No more features to split on
            predicted_feature_values = [example[feature_to_predict] for example in examples]
            most_common_value = max(set(predicted_feature_values), key=predicted_feature_values.count)
            return Tree(value=most_common_value)

        best_feature = self.get_most_informative(info_gains)
        tree = Tree(attribute=best_feature)

        feature_values = set([example[best_feature] for example in examples])
        for value in feature_values:
            filtered_examples = [example for example in examples if example[best_feature] == value]
            if len(filtered_examples) == 0:
                # No examples with this feature value
                predicted_feature_values = [example[feature_to_predict] for example in examples]
                most_common_value = max(set(predicted_feature_values), key=predicted_feature_values.count)
                subtree = Tree(value=most_common_value)
            else:
                updated_gains = info_gains.copy()
                del updated_gains[best_feature]
                subtree = self.build_tree(filtered_examples, updated_gains)
            tree.add_child(value, subtree)

        return tree

# TODO: Temp. gain 0.06423863677845937

if __name__ == '__main__':
    file_path = 'test2.txt'
    p = Prediction(file_path)
    p.make_tree()
    p.tree.print_tree()
    pass

