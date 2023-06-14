import numpy as np

np.random.seed(0)


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
                feature_probabilities[feature][option] = {
                    counts[0] / total_neg,
                    counts[1] / total_pos
                }
        pass

        return label_probabilities, feature_probabilities

    def predict(self, instance):
        predicted_label = None
        max_probability = 0.0

        for label, label_probability in self.label_probabilities.items():
            instance_probability = label_probability
            for feature_index, feature_value in enumerate(instance):
                if (
                        feature_index in self.feature_probabilities
                        and feature_value in self.feature_probabilities[feature_index]
                ):
                    feature_label_probabilities = self.feature_probabilities[feature_index][feature_value]
                    if label in feature_label_probabilities:
                        instance_probability *= feature_label_probabilities[label]

            if instance_probability > max_probability:
                max_probability = instance_probability
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
        self.bayes_model.fit(examples=self.train_examples, labels=self.get_labels(self.train_examples))

    def parse_data(self, txt, test=False):
        with open(txt, 'r') as file:
            lines = file.readlines()
        data = []
        header = lines[0].replace("\n", "").split('\t')
        for line in lines[1:]:
            values = line.replace("\n", "").split('\t')
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
            file.write(str(tree_prediction) + ' ' + str(bayes_prediction) + '\n')
        tree_acc = round(self.get_accuracy(self.predicted["tree"]), 2)
        naive_acc = round(self.get_accuracy(self.predicted["naive"]), 2)
        file.write(str(tree_acc) + ' ' + str(naive_acc) + '\n')

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
        feature_values = set([example[feature] for example in examples])
        for value in feature_values:
            predicted_neg = count_sub_dict[feature][value][0]
            predicted_pos = count_sub_dict[feature][value][1]
            ttl_predicted_neg += predicted_neg
            ttl_predicted_pos += predicted_pos
            feature_entropys[value] = {
                'entropy': self.get_entropy(neg_examples=predicted_neg, pos_examples=predicted_pos),
                'ttl_count_for_option': predicted_neg + predicted_pos
            }
        feature_entropys['ttl_entropy_for_feature'] = self.get_entropy(neg_examples=ttl_predicted_neg,
                                                                       pos_examples=ttl_predicted_pos)
        return feature_entropys

    def get_gain_for_feature(self, entropies_for_feature):
        ttl_feature_gain = entropies_for_feature['ttl_entropy_for_feature']
        for option in entropies_for_feature.values():
            if isinstance(option, dict):
                option_count = option['ttl_count_for_option']
                option_entropy = option['entropy']
                ttl_feature_gain -= (option_count / self.ttl_examples) * option_entropy
        return ttl_feature_gain

    def calc_entropies_for_all(self, features, examples):
        all_entropies = {}
        for feature in features:
            all_entropies[feature] = self.get_entropies_for_feature(feature, examples)
        return all_entropies

    def calc_gain_for_all(self, features, examples):
        new_entropies = self.calc_entropies_for_all(features, examples)
        info_gains = {}
        for feature, entropies in new_entropies.items():
            info_gains[feature] = self.get_gain_for_feature(entropies_for_feature=entropies)
        return info_gains

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
        n_yes = values.count('Yes') + values.count('yes')
        n_no = values.count('No') + values.count('no')
        if n_yes >= n_no:
            return 'Yes'
        else:
            return 'No'

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
        # No more features to split on
        if len(info_gains) == 1:
            predicted_feature_values = [example[self.feature_to_predict] for example in examples]
            most_common_value = self.mode(predicted_feature_values)
            return Tree(value=most_common_value)

        best_feature = self.get_most_informative(info_gains)

        tree = Tree(feature=best_feature)

        feature_values = set([example[best_feature] for example in examples])

        for option in feature_values:
            filtered_examples = [example for example in examples if example[best_feature] == option]
            predicted_feature_values = [example[self.feature_to_predict] for example in examples]
            if len(filtered_examples) == 0:
                # No examples with this feature value
                most_common_value = self.mode(predicted_feature_values)
                subtree = Tree(value=most_common_value)
            else:
                if best_feature in features:
                    features.remove(best_feature)
                updated_gains = self.calc_gain_for_all(features, filtered_examples)
                default = self.mode(predicted_feature_values)
                subtree = self.build_tree(filtered_examples, updated_gains, default, list(features))
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
    train_file = 'test2.txt'
    test_file = 'test2.txt'
    out_tree = 'my_output_tree.txt'
    out = 'my_output.txt'

    train_and_evaluate(train_data=train_file, test_data=test_file, out_tree_file=out_tree, output=out)
