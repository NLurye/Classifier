import math
from collections import OrderedDict

import numpy as np



class NaiveBayes:
    def __init__(self, counts):
        self.label_probabilities = None
        self.feature_probabilities = None
        self.counts_dic = counts

    def fit(self, labels):
        self.label_probabilities, self.feature_probabilities = self.calc_probs(labels)

    def calc_probs(self, labels):
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        label_probabilities = {}
        for label, count in label_counts.items():
            label_probabilities[label] = count / len(labels)

        feature_probabilities = {}
        for feature, options in self.counts_dic.items():
            feature_probabilities[feature] = {}
            total_neg = sum(option[0] for option in options.values())
            total_pos = sum(option[1] for option in options.values())

            for option, counts in options.items():
                # smoothing
                feature_probabilities[feature][option] = {
                    list(label_counts.keys())[0]: (counts[0] + 1) / (total_neg + len(options)),
                    list(label_counts.keys())[1]: (counts[1] + 1) / (total_pos + len(options))
                }

        return label_probabilities, feature_probabilities

    def predict(self, instance):
        predicted_label = None
        max_posteriori = -math.inf

        for label, label_probability in self.label_probabilities.items():
            posteriori = math.log(label_probability)

            for feature_name, feature_value in instance.items():
                if (
                        feature_name in self.feature_probabilities
                        and feature_value in self.feature_probabilities[feature_name]
                ):
                    likelihood = self.feature_probabilities[feature_name][feature_value][label]
                    posteriori += math.log(likelihood)

            if posteriori > max_posteriori:
                max_posteriori = posteriori
                predicted_label = label

        return predicted_label


class Tree:
    def __init__(self, feature=None, value=None):
        self.feature = feature
        self.branch = value
        self.children = {}

    def add_child(self, value, subtree):
        self.children[value] = subtree

    def predict(self, example):
        if self.branch is not None:
            return self.branch  # Reached a leaf, return the predicted value

        feature_value = example[self.feature]
        if feature_value not in self.children:
            return None  # Unknown feature value, return None

        child_subtree = self.children[feature_value]
        return child_subtree.predict(example)

    def print_tree(self, file, indent=''):
        for branch, subtree in self.children.items():
            if indent == '':
                print(indent + self.feature + '=' + str(branch), end="")
                file.write(indent + self.feature + '=' + str(branch))
            else:
                print(indent + '|' + self.feature + '=' + str(branch), end="")
                file.write(indent + '|' + self.feature + '=' + str(branch))
            if subtree.branch in ['Yes', 'yes', 'no', 'No']:
                print(':' + subtree.branch + '\n', end="")
                file.write(':' + subtree.branch + '\n')
            else:
                print('\n', end="")
                file.write('\n')
                subtree.print_tree(file, indent + '\t')


class Classifier:

    def __init__(self, train_data, test_data):
        self.true_labels = []
        # preprocess data
        self.feature_to_predict, self.ttl_examples, self.train_examples = self.parse_data(train_data)
        self.count_dict = self.count_features(self.train_examples)
        self.set_test_data(test_data)
        self.info_gains = None
        self.all_entropies = None
        self.tree = Tree()
        self.bayes_model = NaiveBayes(counts=self.count_dict)
        # train models
        self.make_tree()
        self.bayes_model.fit(labels=self.get_labels(self.train_examples))

    def parse_data(self, txt, test=False):
        with open(txt, 'r') as file:
            lines = file.readlines()
        data = []
        header = lines[0].replace("\n", "").split('\t')
        for line in lines[1:]:
            values = line.replace("\n", "").split('\t')
            values[-1] = values[-1].lower()
            if test:
                self.true_labels.append(values.pop())
            data.append(dict(zip(header, values)))
        return header[-1], len(data), data

    def get_labels(self, examples):
        last_column = []
        for dictionary in examples:
            values = list(dictionary.values())
            last_value = values[-1]  # Extract the last value from the dictionary
            last_column.append(last_value)
        return last_column

    def set_test_data(self, test_file):
        feature_to_predict, ttl_examples, self.test_dict = self.parse_data(test_file, test=True)

    def predict_on_test_data(self, file):
        self.predicted = {"tree": [], "naive": []}
        for example in self.test_dict:
            tree_prediction = self.tree.predict(example)
            bayes_prediction = self.bayes_model.predict(example)
            self.predicted['tree'].append(tree_prediction)
            self.predicted['naive'].append(bayes_prediction)
            file.write(str(tree_prediction) + '\t' + str(bayes_prediction) + '\n')
        tree_acc = round(self.get_accuracy(self.predicted["tree"]), 2)
        naive_acc = round(self.get_accuracy(self.predicted["naive"]), 2)
        file.write(str(tree_acc) + '\t' + str(naive_acc) + '\n')

    '''
    given amounts of neg and pos examples where specific feature = specific option
    return entropy
    '''
    def get_entropy(self, neg_examples, pos_examples):
        ttl_examples = neg_examples + pos_examples
        if neg_examples == 0 or pos_examples == 0:
            return 0.0  # Return 0 entropy for zero examples
        p_neg = neg_examples / ttl_examples  # positive proportion
        p_pos = pos_examples / ttl_examples  # negative proportion
        return - (p_neg * np.log2(p_neg)) - (p_pos * np.log2(p_pos))

    '''
    given feature and list of relevant examples
    count per option neg/pos examples (on predicted feature) and save the entropy
    count once amount of neg/pos examples for feature (all examples) and save the entropy
    return dict[feature_name] -> {'entropy': x, 'ttl_count_for_option': y}
    '''
    def get_entropies_for_feature(self, feature, examples):
        feature_entropys = {}
        ttl_predicted_neg = 0
        ttl_predicted_pos = 0
        count_sub_dict = self.count_features(examples)
        feature_values = set([example[feature] for example in examples])
        for value in feature_values:
            predicted_neg = count_sub_dict[feature][value][0]
            predicted_pos = count_sub_dict[feature][value][1]
            ttl_predicted_neg += predicted_neg
            ttl_predicted_pos += predicted_pos
            feature_entropys[value] = {
                'entropy': self.get_entropy(neg_examples=predicted_neg, pos_examples=predicted_pos),
                'ttl_count_for_option': predicted_neg + predicted_pos  # for future proportion calculation
            }
        feature_entropys['ttl_entropy_for_feature'] = self.get_entropy(neg_examples=ttl_predicted_neg,
                                                                       pos_examples=ttl_predicted_pos)
        feature_entropys['ttl_count_for_feature'] = ttl_predicted_neg + ttl_predicted_pos
        return feature_entropys

    '''
    given dict (for specific feature) with entropy per option and one ttl entropy
    return gain for this feature
    '''
    def get_gain_for_feature(self, entropies_for_feature):
        ttl_feature_gain = entropies_for_feature['ttl_entropy_for_feature']
        for option in entropies_for_feature.values():
            if isinstance(option, dict):
                option_count = option['ttl_count_for_option']
                option_entropy = option['entropy']
                ttl_feature_gain -= (option_count / entropies_for_feature['ttl_count_for_feature']) * option_entropy #TODO:??????????????? update
        return ttl_feature_gain

    def calc_entropies_for_all(self, features, examples):
        all_entropies = {}
        for feature in features:
            all_entropies[feature] = self.get_entropies_for_feature(feature, examples)
        return all_entropies

    '''
    split the entropies dict per feature and for each calculate and save gain
    return dict[feature_name] -> gain
    '''
    def calc_gain_for_all(self, features, examples):
        new_entropies = self.calc_entropies_for_all(features, examples)
        info_gains = {}
        for feature, entropies in new_entropies.items():
            info_gains[feature] = self.get_gain_for_feature(entropies_for_feature=entropies)
        return info_gains

    '''
    Count amount of neg/pos examples per option per feature in given list of examples
    input: examples = list of dict[feature_name][option]
    output: dict[feature_name][option][count_neg_examples,count_pos_examples]
    '''
    def count_features(self, examples):
        feature_options = {}
        for example in examples:
            for feature, option in example.items():
                if feature != self.feature_to_predict:
                    if feature not in feature_options:
                        feature_options[feature] = {}
                    if option not in feature_options[feature]:
                        feature_options[feature][option] = [0, 0]
                    if example[self.feature_to_predict] == 'No' or example[self.feature_to_predict] == 'no':
                        feature_options[feature][option][0] += 1
                    else:
                        feature_options[feature][option][1] += 1
        return feature_options

    def get_most_informative(self, gains):
        return max(gains, key=gains.get)

    def mode(self, values):
        n_yes = values.count('yes')
        n_no = values.count('no')
        if n_yes >= n_no:
            return 'yes'
        else:
            return 'no'

    '''
    once calculate all gains for all examples
    send gains with all features to recursion func to further calculations on subtrees
    '''
    def make_tree(self):
        features = list(self.count_dict.keys())
        self.all_entropies = self.calc_entropies_for_all(features, self.train_examples)
        self.info_gains = self.calc_gain_for_all(features, self.train_examples)
        features = list(self.info_gains.keys())
        self.tree = self.build_tree(self.train_examples, self.info_gains, None, features)

    def build_tree(self, examples, info_gains, default, features):
        # Examples is empty return default
        if len(examples) == 0:
            return default
        # All examples have the same classification - return classification
        if len(set([example[self.feature_to_predict] for example in examples])) == 1:
            return Tree(value=examples[0][self.feature_to_predict])

        best_feature = self.get_most_informative(info_gains) if len(features) > 1 else next(iter(features))

        tree = Tree(feature=best_feature)
        # get options for best feature out of relevant examples (branches)
        feature_values = list(OrderedDict.fromkeys([example[best_feature] for example in examples]))

        for option in feature_values:
            filtered_examples = [example for example in examples if example[best_feature] == option]
            predicted_feature_values = [example[self.feature_to_predict] for example in examples]
            if len(filtered_examples) == 0 or len(features) == 1:
                # No examples for this option or no more features to split on
                most_common_value = self.mode(predicted_feature_values)
                subtree = Tree(value=most_common_value)
            else:
                # remove chosen feature, since deeper in the tree we already know its value (option)
                if best_feature in features:
                    features.remove(best_feature)
                default = self.mode(predicted_feature_values)
                if len(features) > 1:
                    updated_gains = self.calc_gain_for_all(features, filtered_examples)
                    subtree = self.build_tree(filtered_examples, updated_gains, default, list(features))
                else:
                    # Last feature left - no need to recalculate gains
                    subtree = self.build_tree(filtered_examples, info_gains.pop(best_feature), default, list(features))
            tree.add_child(option, subtree)

        return tree

    def get_accuracy(self, predicted_labels):
        correct_count = sum(1 for pred, true in zip(predicted_labels, self.true_labels) if pred == true)
        return correct_count / len(predicted_labels)


def train_and_evaluate(train_data, test_data, out_tree_file, output):
    c = Classifier(train_data, test_data)  # set data for train & test and build both models
    with open(out_tree_file, "w") as file1:  # print tree
        c.tree.print_tree(file=file1)
    with open(output, "w") as file2:  # print predictions of both and accuracy
        c.predict_on_test_data(file=file2)


if __name__ == "__main__":
    train_file = 'tennis_from_book.txt'
    test_file = 'tennis_from_book.txt'
    train_file = 'tennis_from_lecture.txt'
    test_file = 'tennis_from_lecture.txt'
    train_file = 'test.txt'
    test_file = 'test.txt'
    out_tree = 'output_tree.txt'
    out = 'output.txt'
    train_and_evaluate(train_data=train_file, test_data=test_file, out_tree_file=out_tree, output=out)