import numpy as np
np.random.seed(0)

class NaiveBayes:
    def __init__(self):
        pass

    def predict(self, example):
        pass


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
        if self.branch is not None:
            print(indent + self.branch)
            return

        for branch, subtree in self.children.items():
            print(indent + self.feature + ' = ' + str(branch) + ':')
            subtree.print_tree(file, indent + '\t' + '\t\t')


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
        n_yes = values.count('Yes')
        n_no = values.count('no')
        if n_yes >= n_no:
            return 'Yes'
        else:
            return 'no'

    def make_tree(self):
        features = list(self.count_dict.keys())
        self.all_entropies = self.calc_entropies_for_all(features, self.train_examples)
        self.info_gains = self.calc_gain_for_all(features, self.train_examples)
        features = list(self.info_gains.keys())
        self.tree, _ = self.build_tree(self.train_examples, self.info_gains, None, features)

    def build_tree(self, examples, info_gains, default, features):
        # Examples is empty return default
        if len(examples) == 0:
            return default, features
        # All examples have the same classification - return classification
        if len(set([example[self.feature_to_predict] for example in examples])) == 1:
            return Tree(value=examples[0][self.feature_to_predict]), features
        # No more features to split on
        if len(info_gains) == 1:
            predicted_feature_values = [example[self.feature_to_predict] for example in examples]
            most_common_value = self.mode(predicted_feature_values)
            return Tree(value=most_common_value), features

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
                subtree, features = self.build_tree(filtered_examples, updated_gains, default, features)
            tree.add_child(option, subtree)

        return tree, features


if __name__ == '__main__':
    train_file = 'test2.txt'
    p = Classifier(train_file)
    test_file = 'test2.txt'
    p.set_test_data(test_file)
    p.make_tree()
    p.tree.print_tree(file='my_output_tree.txt')
    # p.predict_on_test_data(file='my_output.txt')
    pass

