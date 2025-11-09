import os
from typing import Callable

from game.gameInstance import Actions, GameInstance
from training.decisionTree3 import FeatureSet


class Logger:
    """
    Classe per la creazione di dataset

    Methods
    -------
    record(gi: GameInstance, action:Actions)
        registra le informazioni osservate e l'azione scelta nello stesso contesto
    """
    def __init__(self, filename: str, features: FeatureSet, featuresExtractor: Callable[[GameInstance], dict[str, any]], overwrite: bool = False):
        """
        Costruttore

        Parameters
        ----------
        filename: str
            file su cui salvare il dataset
        features: FeatureSet
            insieme di features. Verranno scritte nell'header del file csv
        featuresExtractor: Callable[[GameInstance], dict[str, any]]
            Le feature da estrarre. I loro valori verranno scritti come righe del file csv
        overwrite: bool = False
            Sovrascrive il file se presente, altrimenti "appende" le righe al file esistente
        """
        self.records = set()
        self.featuresExtractor = featuresExtractor
        self.toCsv = features.toCsv
        # apre il file del dataset appendendo il nuovo contenuto a quello esistente se overwrite = False, altrimenti riscrive tutto il file
        if os.path.isfile(filename) and not overwrite:
            self.file = open(filename, 'r+')
            self.records = set(self.file.readlines())
        else:
            self.file = open(filename, 'w')
            self.file.write(f",{features}\n")

    def __len__(self):
        return len(self.records)

    def record(self, gi: GameInstance, action: Actions):
        """
        Registra le informazioni osservate e l'azione scelta nello stesso contesto

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco
        action: Actions
            azione scelta nel contesto di gi
        """
        record = f"{action.name},{self.toCsv(self.featuresExtractor(gi))}\n"
        if record not in self.records:
            self.records.add(record)
            self.file.write(record)

    def __del__(self):
        self.file.close()
