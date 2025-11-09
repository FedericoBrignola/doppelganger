import multiprocessing

import pandas as pd

from training.decisionTree3 import FeatureSet
from training.id3 import id3


class Id3Generation:
    """
    Genera la prima popolazione usando l'algoritmo ID3
    """
    def __init__(self, trainingSet: pd.DataFrame, features: FeatureSet, populationLen: int):
        """
        Parameters
        ----------
        trainingSet: pd.DataFrame
            dataset per il training
        features: FeatureSet
            l'insieme di features degli alberi
        populationLen: int
            grandezza popolazione
        """
        self.trainingSet = trainingSet
        self.features = features
        self.populationLen = populationLen

    def __call__(self):
        population = []
        with multiprocessing.Pool() as pool:  # usato per la multiprogrammazione
            results = list()
            # lancia un processo che esegue l'ID3 per ogni individuo da generare. Nota che a ognuno viene passato lo 0.7 del trainingset
            for i in range(self.populationLen):
                print("[firstGeneration.Id3Generation][ INFO ] inducing", i)
                results.append(pool.apply_async(id3, [self.trainingSet.sample(frac=0.7), self.features]))
            # recupera i risultati dei vari processi lanciati
            for i, r in enumerate(results):
                population.append(r.get())
                print("[firstGeneration.Id3Generation][ INFO ] end induction of ", i)
        return population
