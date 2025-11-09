from __future__ import annotations
import numpy as np
import pandas as pd
from training.decisionTree3 import DecisionTree, DecisionNode, FeatureSet
from math import inf


def id3(trainingSet: pd.DataFrame, features: FeatureSet) -> DecisionTree:
    """
    Genera un albero decisionale. Viene usata l'entropia come misura per la scelta delle features

    Parameters
    ----------
    trainingSet: pandas.DataFrame
        dataset da cui generare l'albero
    features: FeatureSet
        features dell'albero

    Return
    ------
    DecisionTree: albero generato
    """
    t = DecisionTree(features)
    t.add(DecisionNode())  # Aggiunge il nodo radice all' albero. Il nodo attualmente e' vuoto
    stack = [(0, trainingSet)]  # Lo stack contiene tuple del tipo (idNode, trainingSet) dove trainingSet contiene solo gli esempi che il nodo deve classificare

    while len(stack) > 0:
        nodeId, dataset = stack.pop()
        node = t.nodes.get(nodeId)
        outputDomain = dataset.index  # Ottiene tutte le classificazioni delle istanze in dataset

        if dataset.empty:
            continue

        # se c'e' una sola classificazione, la assegna al nodo
        if outputDomain.unique().size == 1:
            node.out = outputDomain[0]
            continue

        # altrimenti ricava la feature e il threshold migliore secondo l'entropia
        bestFeature, bestValue = bestFeatureByEntropy(dataset, features)
        node.feature = bestFeature

        # usiamo la classificazione piu' ricorrente nel dataset come valore di default per i nodi figli
        default = outputDomain.value_counts().idxmax()

        if bestFeature is None:
            """In caso non si possono piu' effettuare tagli al dataset (il dataset contiene 'rumore' che porta istanze uguali ad avere output diversi)"""
            node.out = default
            continue

        # se la feature selezionata ha un ordinamento, generiamo due nodi figli e dividiamo il dataset secondo i valori minori/maggiori del threshold
        if features[node.feature].isNumerical():
            node.threshold = bestValue
            cD = dataset[dataset[node.feature] < node.threshold]  # cD conterra' solo le istanze con feature < threshold
            cId = t.add(DecisionNode(out=default), nodeId, True)  # aggiungiamo il nodo all'albero
            stack.append((cId, cD))

            cD = dataset[dataset[node.feature] >= node.threshold]  # cD conterra' solo le istanze con feature >= threshold
            cId = t.add(DecisionNode(out=default), nodeId, False)  # aggiungiamo il nodo all'albero
            stack.append((cId, cD))
        # se invece la feature non ha un ordinamento, generiamo un figlio per ogni valore del dominio (osservato) della feature e ripartiamo di conseguenza il dataset
        else:
            for label in dataset[node.feature].unique():  # per ogni valore del dominio (osservato) della feature
                cD = dataset[dataset[node.feature] == label]  # cD conterra' solo le istanze con feature = label
                cId = t.add(DecisionNode(out=default), nodeId, label)  # aggiungiamo il nodo all'albero
                stack.append((cId, cD))
    return t


def bestFeatureByEntropy(dataset: pd.DataFrame, features: FeatureSet) -> tuple[str, int]:
    """
    Seleziona la feature migliore e il suo valore di threshold secondo l'entropia

    Parameters
    ----------
    dataset: pandas.DataFrame
        dataset per calcolare l'entropia
    features: FeatureSet
        features tra cui scegliere

    Return
    ------
    tuple[str, int]: restituisce la coppia (features, threshold)
    """
    N = dataset.index.size  # Numero di esempi nel dataset
    o = dataset.index.value_counts()  # restituisce il numero di ricorrenze di ogni classificazione del dataset
    H = -np.sum([x / N * np.log2(x / N) for x in o])

    bestDeltaH = 0
    bestAttr = None
    bestValue = None
    for attr in dataset.columns:  # per ogni features del dataset
        if features[attr].isNumerical():
            # per ogni valore osservato della feature calcolo l'entropia usando il valore corrente 'i' come threshold
            for i in dataset[attr].unique():
                NiLeft = dataset[dataset[attr] < i].index.size  # numero di istanze con attr < i
                oiLeft = dataset[dataset[attr] < i].index.value_counts()  # numero di ricorrenze di ogni classificazione con attr < i
                HiLeft = -np.sum([x / NiLeft * np.log2(x / NiLeft) for x in oiLeft])

                NiRight = dataset[dataset[attr] >= i].index.size # numero di istanze con attr >= i
                oiRight = dataset[dataset[attr] >= i].index.value_counts()  # numero di ricorrenze di ogni classificazione con attr >= i
                HiRight = -np.sum([x / NiRight * np.log2(x / NiRight) for x in oiRight])

                Hi = (NiLeft / N) * HiLeft + (NiRight / N) * HiRight
                deltaH = H - Hi
                if deltaH > bestDeltaH:  # trovata feature e threshold migliori
                    bestDeltaH = deltaH
                    bestValue = i
                    bestAttr = attr

        else:
            Hattr = 0
            for i in dataset[attr].unique():  # per ogni valore osservato della feature
                Ni = dataset[dataset[attr] == i].index.size  # numero di istanze con attr = i
                oi = dataset[dataset[attr] == i].index.value_counts() # numero di ricorrenze di ogni classificazione con attr = i
                Hi = -np.sum([x / Ni * np.log2(x / Ni) for x in oi])
                Hattr += (Ni / N) * Hi
            deltaH = H - Hattr
            if deltaH > bestDeltaH:  # trovata feature e threshold migliori
                bestDeltaH = deltaH
                bestAttr = attr
    return bestAttr, bestValue