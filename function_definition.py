from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes, BayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sys import version_info

if version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


def discretize(df: 'pd.DataFrame', columns_to_discretize: 'list[str]', bins_edges: 'dict',
               categories: 'dict' = None) -> pd.DataFrame:
    """

    :param df: pandas dataframe
    :param columns_to_discretize: List of the features that you want to discretize
    :param bins_edges: Dictionary containing for each feature the edges of the bins
    :param categories: Dictionary containing for each feature the name of each bin
    :return: A copy of the discretized dataframe
    """
    new_df = df.copy()
    for col in columns_to_discretize:
        df[col] = pd.cut(df[col], bins_edges[col], labels=categories[col])

    return df


def get_node_size(label: 'str') -> float:
    """

    :param label: Name of the node.
    :return: The necessary dimensions of a circle for the lable to be inside the circle
    """

    fig, ax = plt.subplots(figsize=(3, 3))
    text = ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=12)
    bbox = text.get_window_extent()
    width, height = bbox.width, bbox.height
    plt.close(fig)
    return 250 + max(width, height) * 40


def draw_net(nodes: 'list[str]', net: 'BayesianNetwork|NaiveBayes',
             style: 'Literal["full","trimmed","circular"]' = 'full', target=None) -> None:
    """

    :param nodes: List of nodes sorted in drawing order
    :param net: Bayesian network
    :param style: Indicates the style used to draw our net in the best way. It could be 'full', 'trimmed' or 'circular'
    :param target: In case style='circular' the node 'target' will be positioned in te center of the circle
    :return: The function hasn't a return value. It displays the structure of the net
    """

    graph = nx.DiGraph(net.edges())
    node_sizes = [get_node_size(label) for label in graph.nodes]

    plt.figure(figsize=(15, 10))
    pos = {}

    if style == 'full':
        for i in range(len(graph.nodes)):
            if i < 2:
                pos[nodes[i]] = [i, 4]
            elif i < 10:
                pos[nodes[i]] = [(i - 5), 3]
            elif i < 14:
                pos[nodes[i]] = [(i - 10), 2]
            elif i < 21:
                pos[nodes[i]] = [(i - 15), 1]
            else:
                pos[nodes[i]] = [(i - 18), 0]

    elif style == 'trimmed':
        for i in range(len(graph.nodes)):
            if i < 1:
                pos[nodes[i]] = [i, 4]
            elif i < 9:
                pos[nodes[i]] = [(i - 5), 3]
            elif i < 13:
                pos[nodes[i]] = [(i - 10), 2]
            elif i < 18:
                pos[nodes[i]] = [(i - 15), 1]
            else:
                pos[nodes[i]] = [(i - 18), 0]

    else:
        pos = nx.circular_layout(graph, center=(0, 0))
        pos[target] = (-0.01, 0)

    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.5, arrows=True, arrowstyle='->,head_width=0.1,head_length=2')
    nx.draw_networkx_labels(graph, pos, font_color='black', font_size=10)

    plt.show()


def get_the_posterior_probability(model: 'VariableElimination', data: 'pd.DataFrame', target: 'str') -> list:
    """

    :param model: VariableElimination obtained by a Bayesian Net
    :param data: Dataframe containing the data
    :param target: Variable to predictt
    :return: A list of DiscreteFactor containing the posterior probability associated ad every datapoint
    """

    result = []
    for _, datapoint in data.iterrows():
        row = datapoint.to_dict()
        posterior_probs = model.query(
            variables=[target],
            evidence=row,
            show_progress=False
        )
        result.append(posterior_probs)
    return result


def classify(posterior_prob_tabs: 'list[DiscreteFactor]') -> list:
    """

    :param posterior_prob_tabs: List of posterior probability tables
    :return: A list of predicted labels. Each
            datapoint are classified to the element with a maximum posterior probability
    """

    predicted_labels = []
    for ppt in posterior_prob_tabs:
        predicted_labels.append(ppt.state_names[ppt.variables[0]][np.argmax(ppt.values)])

    return predicted_labels


def compare(y_true: 'list', y_computed: 'list',
            scoring_function: 'Literal["accuracy","recall", "precision", "f1_score"]',
            average: 'Literal["macro", "disjointed"]' = 'macro') -> float | object:


    if not len(y_true) == len(y_computed):
        raise Exception('y_true and y_computed must be the same length.')
    if average != 'macro' and scoring_function == 'accuracy':
        raise Exception('accuracy can only work with macro.')
    if scoring_function == 'accuracy':
        return compute_accuracy(y_true, y_computed)
    if scoring_function == 'recall':
        return compute_recall(y_true, y_computed, average)
    if scoring_function == 'precision':
        return compute_precision(y_true, y_computed, average)
    if scoring_function == 'f1_score':
        return compute_f1_score(y_true, y_computed, average)


def compute_accuracy(y_true: 'list', y_computed: 'list') -> float:
    nominator = 0
    for i in range(len(y_true)):
        if y_true[i] == y_computed[i]:
            nominator += 1

    return nominator / len(y_true)


def compute_f1_score(y_true: 'list', y_computed: 'list', average: 'Literal["macro", "disjointed"]') -> object:
    precision = compute_precision(y_true, y_computed, average='disjointed')
    recall = compute_recall(y_true, y_computed, average='disjointed')

    unique_values = np.unique(y_true)

    f1_score = {}

    for value in unique_values:
        nominator = precision[value] * recall[value]
        denominator = precision[value] + recall[value]

        f1_score[value] = 2 * nominator / denominator

    if average == 'macro':
        return np.mean([f1_score[l] for l in f1_score])
    return f1_score


def compute_precision(y_true: 'list', y_computed: 'list', average: 'Literal["macro", "disjointed"]') -> object:
    unique_values = np.unique(y_true)
    accuracy = {}
    for value in unique_values:

        nominator = 0
        denominator = 0
        for i in range(len(y_true)):
            if y_computed[i] == value:
                if y_true[i] == y_computed[i]:
                    nominator += 1
                denominator += 1

        accuracy[value] = nominator / denominator
    if average == 'macro':
        return np.mean([accuracy[l] for l in accuracy])
    return accuracy


def compute_recall(y_true: 'list', y_computed: 'list', average: 'Literal["macro", "disjointed"]') -> object:
    unique_values = np.unique(y_true)
    recall = {}
    for value in unique_values:

        nominator = 0
        denominator = 0
        for i in range(len(y_true)):
            if y_computed[i] == value:
                if y_true[i] == y_computed[i]:
                    nominator += 1
            if y_computed[i] != value and y_true[i] == value:
                denominator += 1

        recall[value] = nominator / (denominator + nominator)
    if average == 'macro':
        return np.mean([recall[l] for l in recall])
    return recall
