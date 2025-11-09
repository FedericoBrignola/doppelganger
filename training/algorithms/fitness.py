import pandas as pd

from training.decisionTree3 import DecisionTree


class AccuracyBasedFitness:
    """
    Fitness dato dal rapporto (classificazioniCorrette / classificazioniTotali)
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def __call__(self, elem: DecisionTree):
        c = 0  # contatore delle classificazioni corrette
        for i, row in self.dataset.iterrows():
            if elem.predict(row.to_dict()) == i:
                c += 1
        acc = c / self.dataset.index.size  # classificazioni corrette su istanze totali
        return acc


class AccuracyBasedFitnessPwm2(AccuracyBasedFitness):
    """
    Fitness dato dal rapporto (classificazioniCorrette / classificazioniTotali)^2
    """
    def __call__(self, elem: DecisionTree):
        acc = super().__call__(elem)
        return acc ** 2


class AccuracyBasedFitnessPwm3(AccuracyBasedFitness):
    """
    Fitness dato dal rapporto (classificazioniCorrette / classificazioniTotali)^3
    """
    def __call__(self, elem: DecisionTree):
        acc = super().__call__(elem)
        return acc ** 3


class WeightedFormula:
    """
    Fitness dato dal rapporto [(alpha * accuratezzaSulTesting^2) + (beta * accuratezzaSulTraining^2)]
    """
    def __init__(self, alpha, beta, testing, training):
        self.alpha = alpha
        self.beta = beta
        self.testing = testing
        self.training = training

    def __call__(self, elem: DecisionTree):
        fTesting = AccuracyBasedFitnessPwm2(self.testing)(elem)
        fTraining = AccuracyBasedFitnessPwm2(self.training)(elem)
        return (self.alpha * fTesting + self.beta * fTraining) / (self.alpha + self.beta)


class RarityBasedFitness:
    """
        Il fitness e' basatto sulla WeightedFormula dove i pesi sono basati sulla rarita' delle label. Piu' una label
        e' rara, maggiore sara' il suo peso nella formula.
        Formula:
            per ogni label L, sia:
                - c il numero di istanze classificate correttamente in L
                - T il numero totale di istanze con label L
                - p il rapporto tra X e le istanze totali del dataset
            Sum (1-p)*(c/T)

            Il risultato sara' poi normalizzato
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        perc = dataset.index.value_counts(normalize=True)
        self.classifications = {}
        for label in perc.index:
            self.classifications[label] = 1 - perc[label]

    def __call__(self, elem: DecisionTree):
        rightClassifications = dict()
        totalClassifications = dict()
        for label in self.classifications.keys():
            rightClassifications[label] = 0
            totalClassifications[label] = 0

        for i, row in self.dataset.iterrows():
            if elem.predict(row.to_dict()) == i:
                rightClassifications[i] += 1
            totalClassifications[i] += 1
        fitness = 0
        normalize = 0
        for label, multiplier in self.classifications.items():
            fitness += multiplier*rightClassifications[label]/totalClassifications[label]
            normalize += multiplier
        return fitness/normalize
