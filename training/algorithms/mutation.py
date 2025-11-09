import copy
import random

import pandas as pd

from training.decisionTree3 import DecisionTree
from training.algorithms.pruning import pruning


class ChangeTestMutation:
    def __init__(self):
        pass

    def __call__(self, decisionTree: DecisionTree):
        ids = list(decisionTree.nodes.keys())
        nodeId = random.choice(ids)
        node = decisionTree.nodes[nodeId]
        if node.isLeaf() or not decisionTree.features[node.feature].isNumerical():
            return None
        node.threshold += decisionTree.features[node.feature].getRand()
        pruning(decisionTree, nodeId)

        return nodeId


class ChangeAttributeMutation:
    def __init__(self):
        pass

    def __call__(self, decisionTree: DecisionTree):
        nodeId = random.choice(list(decisionTree.nodes.keys()))
        node = decisionTree.nodes[nodeId]
        newFeature = random.choice(list(decisionTree.features.keys()))
        if node.isLeaf() or not decisionTree.features[node.feature].isNumerical() or \
                not decisionTree.features[newFeature].isNumerical():
            return
        # domainFeature = dataset[newFeature].unique()
        node.feature = newFeature
        node.threshold = decisionTree.features[node.feature].getRand()
        pruning(decisionTree, nodeId)
        return nodeId


class NodeSwapsMutation:
    def __init__(self):
        pass

    def __call__(self, decisionTree: DecisionTree):
        nodes = []
        while len(nodes) < 2:
            nodeId = random.choice(list(decisionTree.nodes.keys()))
            node = decisionTree.nodes[nodeId]
            if not node.isLeaf():
                nodes.append((nodeId, node))

        nodeIdA, nodeA = nodes.pop()
        nodeIdB, nodeB = nodes.pop()
        copyNodeA = copy.deepcopy(nodeA)

        nodeA.setParent(nodeB.parent, nodeB.label)
        nodeA.children = copy.deepcopy(nodeB.children)
        for child in nodeB.children:
            child.parent = nodeIdA

        nodeB.setParent(copyNodeA.parent, copyNodeA.label)
        nodeB.children = copyNodeA.children
        for child in copyNodeA.children:
            child.parent = nodeIdB

        return nodeIdA, nodeIdB
