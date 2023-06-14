import numpy as np
np.random.seed(0)

class NaiveBayes:
    def __init__(self):
        self.label_probabilities = None
        self.feature_probabilities = None

    def fit(self, features, labels):
        self.label_probabilities, self.feature_probabilities = self.naive_bayes(features, labels)

    def predict(self, instance):
        return self.classify_naive_bayes(instance)

    def naive_bayes(self, features, labels):
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
        num_features = len(features[0])
        for feature_index in range(num_features):
            feature_probabilities[feature_index] = {}
            unique_feature_values = set(row[feature_index] for row in features)
            for feature_value in unique_feature_values:
                feature_probabilities[feature_index][feature_value] = {}
                for label in label_counts:
                    feature_label_count = sum(
                        1 for row, classification in zip(features, labels)
                        if row[feature_index] == feature_value and classification == label)
                    feature_label_probability = feature_label_count / label_counts[label]
                    feature_probabilities[feature_index][feature_value][label] = feature_label_probability

        return label_probabilities, feature_probabilities

    def classify_naive_bayes(self, instance):
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

    def __init__(self, train_data):
        self.info_gains = None
        self.all_entropies = None
        self.feature_to_predict, self.ttl_examples, self.train_examples = self.parse_data(train_data)
        self.count_dict = self.count_features(self.train_examples)
        self.tree = Tree()
        self.bayes_model = NaiveBayes()

    def parse_data(self, txt, test=False):
        with open(txt, 'r') as file:
            lines = file.readlines()
        data = []
        header = lines[0].replace("\n", "").split('\t')
        for line in lines[1:]:
            values = line.replace("\n", "").split('\t')
            if test:
                values.pop()
            data.append(dict(zip(header, values)))
        return header[-1], len(data), data

    def set_test_data(self, test_file):
        feature_to_predict, ttl_examples, self.test_dict = self.parse_data(test_file, test=True)

    def predict_on_test_data(self, file):
        with open(file, 'w') as f:
            for example in self.test_dict:
                tree_prediction = self.tree.predict(example)
                bayes_prediction = self.bayes_model.predict(example)
                f.write(str(tree_prediction) + ' ' + str(bayes_prediction) + '\n')

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

    def get_accuracy(self, predicted_labels, true_labels):
        correct_count = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
        return correct_count / len(predicted_labels)

    def evaluate(self, file):
        true_labels = [example[self.feature_to_predict] for example in self.test_dict]

        # Evaluate decision tree
        tree_predictions = [self.tree.predict(example) for example in self.test_dict]
        tree_accuracy = self.get_accuracy(tree_predictions, true_labels)

        # Fit Naive Bayes model
        self.bayes_model.fit(self.train_examples, self.feature_to_predict)

        # Evaluate Naive Bayes
        bayes_predictions = [self.bayes_model.predict(example) for example in self.test_dict]
        bayes_accuracy = self.get_accuracy(bayes_predictions, true_labels)

        # Write predictions and accuracies to file
        with open(file, 'w') as f:
            for tree_pred, bayes_pred in zip(tree_predictions, bayes_predictions):
                f.write(f"{tree_pred}\t{bayes_pred}\n")
            f.write(f"Decision Tree Accuracy: {tree_accuracy}\n")
            f.write(f"Naive Bayes Accuracy: {bayes_accuracy}\n")
def tree_train_and_evaluate():
    pass
if __name__ == "__main__":
    train_file = 'test2.txt'
    test_file = 'test2.txt'
    p = Classifier(train_file)
    p.set_test_data(test_file)
    p.make_tree()
    with open('my_output_tree.txt', "w") as file:
        p.tree.print_tree(file=file)
    p.predict_on_test_data(file='my_output.txt')
    # label_probabilities, feature_probabilities = p.naive_base(features, feature_names, labels)
    # predicted_label = p.classify_naive_base(label_probabilities, feature_probabilities, p.test_dict)
    p.evaluate(output_prediction_path)
