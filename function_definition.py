from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes, BayesianNetwork

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def discretize(df, columns_to_discretize, bins_edges, categories=None):
    """

    :param df: pandas dataframe
    :param columns_to_discretize: List of the features that you want to discretize
    :param bins_edges: Dictionary containing for each feature the edges of the bins
    :param categories: Dictionary containing for each feature the name of each bin
    :return: A copy of the discretized dataframe
    """

    for col in columns_to_discretize:
        df[col] = pd.cut(df[col], bins_edges[col], labels=categories[col])

    return df


def get_node_size(label: 'str'):
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


def draw_net(nodes: 'list[str]', net: 'BayesianNetwork|NaiveBayes', style='full', target=None):
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


def get_the_posterior_probability(model: 'VariableElimination', data: 'pd.DataFrame', target: 'str'):
    """

    :param model: VariableElimination obtained bya Bayesian Net
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


def classify(posterior_prob_tabs: 'lisr[DiscreteFactor]'):
    """

    :param posterior_prob_tabs: List of posterior probability tables
    :return: A list of predicted labels. Each
            datapoint are classified to the element with a maximum posterior probability
    """

    predicted_labels = []
    for ppt in posterior_prob_tabs:
        predicted_labels.append(ppt.state_names[ppt.variables[0]][np.argmax(ppt.values)])

    return predicted_labels
