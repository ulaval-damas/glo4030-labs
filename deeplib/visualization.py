'''
From https://discuss.pytorch.org/t/print-autograd-graph/692/16
'''
from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_vizualization_autograd(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """
    #param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                # node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                node_name = '%s\n %s' % ("Var", size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(var)), str(id(u[0])))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)

    return dot
