import matplotlib.pyplot as plt
from pandas import DataFrame
from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes, BayesianNetwork
import networkx as nx


def get_node_size(label:'str'):
    '''
    returns the necessary dimensions of a circle for the lable to be inside the circle
    '''
    fig, ax = plt.subplots(figsize=(3, 3))
    text = ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=12)
    bbox = text.get_window_extent()
    width, height = bbox.width, bbox.height
    plt.close(fig)
    return 250 + max(width, height) * 40


def draw_net(nodes:'list[str]', net:'BayesianNetwork|NaiveBayes'):
    '''
    draw the network passed as imput creating a big-enough circle for every node so that the text is inside the circle 
    '''
    graph = nx.DiGraph(net.edges())
    node_sizes = [get_node_size(label) for label in graph.nodes]

    pos = {}
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

    plt.figure(figsize=(15,10))
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.5, arrows=True, arrowstyle='->,head_width=0.1,head_length=2')
    nx.draw_networkx_labels(graph, pos, font_color='black', font_size=10)


    plt.show()


def predict(model: 'VariableElimination', data: 'DataFrame', target: 'str'):
    '''
    given a Variable elimination model, a dataframe and a target, returns a list with all the prediction of the target for every data point in the data. 
    '''
    result = []
    for _, datapoint in data.iterrows():
        row = datapoint.to_dict()
        row_result = model.query(
            variables=[target],
            evidence=row,
            show_progress=False
        )
        result.append(row_result)

    return result