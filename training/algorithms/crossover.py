import copy
import random

import pandas as pd

from training.algorithms.pruning import pruning
from training.decisionTree3 import DecisionTree


class SemanticSubtreeCrossover:
    """
    Semantic-based subtrees swapping crossover, common path
    """

    def __init__(self, trainingSet: pd.DataFrame):
        self.trainingSet = trainingSet

    def __call__(self, parents: list[DecisionTree]) -> list[DecisionTree]:
        a = copy.deepcopy(parents[0])  # todo: necessario il deepcopy?
        b = copy.deepcopy(parents[1])
        randRow = self.trainingSet.sample(1).squeeze().to_dict()  # seleziona randomicamente un'istanza del dataset
        pathA: list[int] = [0]
        pathB: list[int] = [0]
        # ottiene il cammino del primo albero che classifica l'istanza selezionata
        while pathA[-1] is not None:
            pathA.append(a.queryNode(randRow, pathA[-1]))
        # ottiene il cammino del secondo albero che classifica l'istanza selezionata
        while pathB[-1] is not None:
            pathB.append(b.queryNode(randRow, pathB[-1]))
        pathA.pop()
        pathB.pop()
        nodeIdA = random.choice(pathA)  # sceglie un nodo casuale del primo cammino
        nodeIdB = random.choice(pathB)  # sceglie un nodo casuale del secondo cammino
        subTreeA = a.extract(nodeIdA)   # estrae il sottoalbero radicato nel primo nodo scelto
        subTreeB = b.extract(nodeIdB)   # estrae il sottoalbero radicato nel secondo nodo scelto
        appendedIdA = a.substitute(nodeIdA, subTreeB)  # sostituisce il sottoalbero del primo nodo con quello del secondo
        appendedIdB = b.substitute(nodeIdB, subTreeA)  # sostituisce il sottoalbero del secondo nodo con quello del primo
        pruning(a, appendedIdA)  # pruning primo albero
        pruning(b, appendedIdB)  # pruning secondo albero
        return [a, b]


class SemanticSubtreeCrossoverSameFeature:
    """
    Semantic-based subtrees swapping crossover, common path chose nodes with same feature
    """

    def __init__(self, trainingSet: pd.DataFrame):
        self.trainingSet = trainingSet

    def __call__(self, parents: list[DecisionTree]) -> list[DecisionTree]:
        a = copy.deepcopy(parents[0])
        b = copy.deepcopy(parents[1])
        randRow = self.trainingSet.sample(1).squeeze().to_dict()
        # i cammini dei due alberi vengono salvati su una struttura a dizionario con chiave: featureName e valore: lista di nodi con quella feature
        # in questo modo possiamo accedere rapidamente ai nodi che hanno una determinata feature
        walkA: dict[str, list[int]] = dict()
        walkB: dict[str, list[int]] = dict()

        # ottiene i nodi del primo cammino
        cnodeId = 0
        while cnodeId is not None:
            f = a.nodes[cnodeId].feature
            if f not in walkA:
                walkA[f] = [cnodeId]
            else:
                walkA[f].append(cnodeId)
            cnodeId = a.queryNode(randRow, cnodeId)

        # ottiene i nodi del secondo cammino
        cnodeId = 0
        while cnodeId is not None:
            f = b.nodes[cnodeId].feature
            if f in walkA:
                if f not in walkB:
                    walkB[f] = [cnodeId]
                else:
                    walkB[f].append(cnodeId)
            cnodeId = b.queryNode(randRow, cnodeId)

        # seleziona una feature randomicamente. Nota, in walkB ci sono solo features presenti anche in walkA,
        # non e' quindi necessario nessun ulteriore controllo
        # scelta la feature, sceglie casualmente un nodo con quella feature nei due cammini e scambia i sottoalberi radicati in questi nodi
        feature = random.choice(list(walkB.keys()))
        nodeIdA = random.choice(walkA[feature])
        nodeIdB = random.choice(walkB[feature])
        subTreeA = a.extract(nodeIdA)
        subTreeB = b.extract(nodeIdB)
        appendedIdA = a.substitute(nodeIdA, subTreeB)
        appendedIdB = b.substitute(nodeIdB, subTreeA)
        pruning(a, appendedIdA)
        pruning(b, appendedIdB)
        return [a, b]


class RandomNodeCrossover:
    @staticmethod
    def __call__(parents: list[DecisionTree]) -> list[DecisionTree]:
        a = copy.deepcopy(parents[0])
        b = copy.deepcopy(parents[1])

        ia = random.choice(list(a.nodes.keys()))
        na = a.nodes[ia]
        while na.isLeaf():
            ia = random.choice(list(a.nodes.keys()))
            na = a.nodes[ia]

        ib = random.choice(list(b.nodes.keys()))
        nb = b.nodes[ib]
        while not (not nb.isLeaf() and a.features[na.feature].isNumerical() == b.features[nb.feature].isNumerical()):
            ib = random.choice(list(b.nodes.keys()))
            nb = b.nodes[ib]

        tempNode = copy.deepcopy(na)
        na.feature = nb.feature
        na.threshold = nb.threshold
        na.out = nb.out
        nb.feature = tempNode.feature
        nb.threshold = tempNode.threshold
        nb.out = tempNode.out

        a.nodes[ia] = na
        b.nodes[ib] = nb

        pruning(a, ia)
        pruning(b, ib)
        return [a, b]


class TestingCrossover:
    """
    [ WARNING ] only for testing purposes
    """

    @staticmethod
    def __call__(parents: list[DecisionTree]) -> list[DecisionTree]:
        a = copy.deepcopy(parents[0])
        b = copy.deepcopy(parents[1])
        ia = random.choice(list(a.nodes.keys()))
        ib = random.choice(list(b.nodes.keys()))
        a.nodes[ia].threshold = 0
        b.nodes[ib].threshold = 0
        pruning(a, ia)
        pruning(b, ib)
        return [a, b]
