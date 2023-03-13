from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes, BayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.estimators import BayesianEstimator

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
        new_df[col] = pd.cut(new_df[col], bins_edges[col], labels=categories[col])

    return new_df


def get_node_size(label: 'str') -> float:
    """

    :param label: Name of the node.
    :return: The necessary dimensions of a circle for the label to be inside the circle
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
    :param target: Variable to predict
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


def my_fit(net: 'BayesianNetwork|NaiveBayes', data: 'pd.DataFrame') -> None:
    """

    :param net: Untrained bayesian network
    :param data: Dataframe used to estimate the parameter of the net
    :return: None
    """
    estimator = BayesianEstimator(net, data)

    pseudo_counts = {}
    for node in data.columns:
        node_card = len(data.loc[:, node].unique())
        parents_card = int(np.prod([len(data.loc[:, p].unique()) for p in net.get_parents(node)]))
        pseudo_counts[node] = np.ones((node_card, parents_card))

    cpds = estimator.get_parameters(prior_type="dirichlet", pseudo_counts=pseudo_counts)

    for cpd in cpds:
        net.add_cpds(cpd)


def check_consistency(models: 'dict [str,BayesianNetwork|NaiveBayes]'):
    """
    This function check the consistency of the models
    :param models: dict of the models to test
    :return: None
    """
    for net_type, net in models.items():
        if net.check_model():
            print(net_type, "consistency check successfully completed")
        else:
            print(net_type, "consistency check failed")


def classify(net: 'BayesianNetwork|NaiveBayes', data: 'pd.DataFrame') -> list:
    """

    :param net: Trained bayesian network that you want to use to predict your data
    :param data: Dataframe that you predict
    :return: A list containing the predicted labels
    """
    predicted_probabilities = net.predict_probability(data)
    predicted_index = predicted_probabilities.idxmax(axis=1)
    y_pred = [predicted_probabilities.columns.get_loc(idx) for idx in predicted_index]

    return y_pred


def multi_bar_plot(data_to_plot: 'list[list]', names: 'list[str]', x_label: 'str', hspace:'float|None' = None) -> 'None':
    """

    :param data_to_plot:
    :param names:
    :param x_label:
    :return:
    """
    fig, axes = plt.subplots(len(data_to_plot), 1, figsize=(15, 5 * len(data_to_plot)))
    axes = axes.flatten()
    if not hspace is None:
        fig.subplots_adjust(hspace=hspace)

    for ax, res, name in zip(axes, data_to_plot, names):
        ax.set_title(f"{name}")
        probs = pd.DataFrame(res)
        probs.plot.bar(x=x_label, ax=ax)

def mp(string:'str') -> 'str':
    """mp, make printable, convert an incoming string to a printable one.

    Args:
        string (str): string to convert

    Returns:
        str: converted string
    """
    return string.replace(" ", "\n").lower()

def get_trimmed_shape(dataframes: 'list[pd.DataFrame]', columns_to_trim:'list[str]', result_dataframe_column_names:'list[str]', result_dataframe_indeces_names:'list[str]') -> 'pd.DataFrame':
    """return a dataframe with the shape of the trimmed dataframe without dupicate on columns columns_to_trim

    Args:
        dataframes (list[pd.DataFrame]): list of dataframe to have get trimmed shape
        columns_to_trim (list[str]): list of colums where to search duplicates
        column_names (list[str]): name of columns of the resulting dataframe
        indeces_names (list[str]): list of indeces of rows

    Returns:
        pd.DataFrame: output dataframe
    """
    res = {}

    for df, idx in zip(dataframes, result_dataframe_indeces_names):
        for col, col_name in zip(columns_to_trim, result_dataframe_column_names):
            df_shape = df[col].drop_duplicates().shape
            shape = f'rows = {df_shape[0]}, columns = {df_shape[1]}'
            res[col_name] = res.get(col_name, []) + [shape]

        res['idx'] = res.get('idx', []) + [idx]

    return pd.DataFrame(res).set_index('idx')