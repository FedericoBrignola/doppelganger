from __future__ import annotations
import copy
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Callable

import pandas
import pygame

from game import gameEngine
from game.gameInstance import GameInstance
from game.gui import Gui
from player.playerProtocol import PlayerI
from player.treeAgent import TreeAgent
from training.decisionTree3 import FeatureSet, DecisionTree
from training.geneticAlgorithm import GA, Pack
from training.logger import Logger

_DATASET_NAME = "dataset.csv"
_ENV_FILE = "env"


@dataclass(init=False)
class Environment:
    """
    Interfaccia di configurazione di un ambiente.

    Methods
    -------
    setFeatures(self, features: FeatureSet, extractor: Callable[[GameInstance], dict[str, any]]) -> None
        Configura i dati delle features
    getFeatures(self) -> FeatureSet
        Restituisce l'insieme di features
    setGameEnv(self, gameInstance: GameInstance) -> None
        Configura l'ambiente di gioco
    newGame(self) -> GameInstance
        Restituisce una nuova istanza di gioco
    setTrainer(self, trainer: PlayerI, maxDataSize: int = None) -> None
        Configura il trainer
    setAlgorithm(self, algorithm: Callable[[pandas.DataFrame], Pack]) -> None
        Configura i dati dell'algoritmo genetico
    """
    # TODO: aggiungi variabile "descrizione": puo' essere utile mostrare la descrizione su un'interfaccia grafica

    trainer: PlayerI                                       # Agente utilizzato per creare il dataset
    features: FeatureSet                                   # L'insieme delle features
    extractor: Callable[[GameInstance], dict[str, any]]    # Funzione che dato uno stato del mondo restituisce dizionario {feature: value}. Sostanzialmente corrisponde ai "sensori" dell'agente
    gameEnv: GameInstance                                  # stato del mondo
    prepackAlgorithm: Callable[[pandas.DataFrame], Pack]   # Funzione che, dato un dataset, restituisce la configurazione dell'algoritmo genetico
    maxDataSize: int                                       # grandezza massima del dataset (numero di righe)

    def __init__(self):
        self.maxDataSize = None
        self.trainer = None
        self.features = None
        self.extractor = None
        self.gameEnv = None
        self.prepackAlgorithm = None

    # Features
    def setFeatures(self, features: FeatureSet, extractor: Callable[[GameInstance], dict[str, any]]) -> None:
        """
        Configura i dati delle features

        Parameters
        ----------
        features: FeatureSet
            l'insieme di features
        extractor: Callable[[GameInstance], dict[str, any]]
            la funzione per estrarre i dati da uno stato del mondo
        """
        if self.features is not None and self.features != features:
            logging.error("It's not a good idea to change the features. Aborting...")
            return
        self.features = features
        self.extractor = extractor

    def getFeatures(self) -> FeatureSet:
        """
        Restituisce l'insieme di features
        """
        return self.features

    def setGameEnv(self, gameInstance: GameInstance) -> None:
        """
        Configura l'ambiente di gioco
        """
        if self.gameEnv is not None and self.gameEnv != gameInstance:
            logging.error("It's not a good idea to change the game environment. Aborting...")
            return
        self.gameEnv = gameInstance

    def newGame(self) -> GameInstance:
        """
        Restituisce una nuova istanza di gioco con le stesse configurazioni del 'gameEnv' se presente, altrimenti con i valori di default
        """
        if self.gameEnv is None:
            logging.error("The game environment has not been set. Returning the default one...")
            return GameInstance()
        return copy.deepcopy(self.gameEnv)

    def setTrainer(self, trainer: PlayerI, maxDataSize: int = None) -> None:
        """
        Configura il trainer

        Parameters
        ----------
        trainer: PlayerI
            L'agente che fara' da trainer
        maxDataSize: int = None
            La grandezza massima del dataset (numero di righe)
        """
        self.maxDataSize = maxDataSize
        self.trainer = trainer

    def setAlgorithm(self, algorithm: Callable[[pandas.DataFrame], Pack]) -> None:
        """
        Configura i dati dell'algoritmo genetico

        Parameters
        ----------
        algorithm: Callable[[pandas.DataFrame], Pack]
            Funzione che, dato un dataset, restituisce un oggetto Pack
        """
        self.prepackAlgorithm = algorithm


class Console:
    """
    Interfaccia per operare su un dato ambiente

    Methods
    -------
    runAlgorithm(self)
        Esegue l'algoritmo genetico
    train(self, maxTimeSeconds: int = 10, overwrite: bool = False)
        Esegue delle partite per creare il dataset
    play(self, player: PlayerI, gui: bool = True, log: bool = False)
        Esegue una partita con un dato giocatore
    replay(self, first: int = 0, last: int = None)
        Effettua delle partite con il miglior individuo di ogni generazione prodotta dal GA
    printDecisionTree(cls, treeFile: str, outFile: str)
        Disegna un DecisionTree
    drawHistoryGraph(self, outFile: str = None)
        Disegna un line chart con l'andamento del fitness nel corso delle generazioni
    """

    HISTORY_GRAPH_FILE = 'hystoryGraph'

    def __init__(self, location: str, env: Environment = None, override: bool = False):  # todo: parametro non utilizzato
        """
        Parameters
        ----------
        location: str
            path usata come rootDir in cui verranno salvati tutti i file relativi al GA
        env: Environment = None
            l'environment da utilizzare
        """
        path = pathlib.Path(location)
        if path.is_file():
            path = path.parent
        elif not path.is_dir():
            os.makedirs(path)

        self.rootDir = path
        self.env = env
        self.gaInstance = None
        self.dataset = None
        datasetPath = os.path.join(self.rootDir, _DATASET_NAME)
        if os.path.isfile(datasetPath):
            self.dataset = pandas.read_csv(datasetPath, index_col=0)
            self.gaInstance = GA(self.env.prepackAlgorithm(self.dataset), self.rootDir)

    def runAlgorithm(self):
        """
        Esegue l'algoritmo genetico
        """
        if self.gaInstance is None:
            if self.dataset is None:
                print("[environment.Console][ ERROR ] Can't run the algorithm without a dataset. Run training first")
                return
            else:
                self.gaInstance = GA(self.env.prepackAlgorithm(self.dataset), self.rootDir,)
        self.gaInstance.run()

    def train(self, maxTimeSeconds: int = 0, overwrite: bool = False, gui: bool = False):
        """
        Esegue delle partite per creare il dataset

        Parameters
        ----------
        maxTimeSeconds: int = 0
            il tempo massimo in cui puo' durare il training (in secondi). Se uguale a 0, non c'e' limite di tempo
        overwrite: bool = False
            Se True, sovrascrive un eventuale dataset gia' esistente, altrimenti lo amplia
        gui: bool = False
           Se true, mostra l' interfaccia di gioco
        """
        if gui:
            Gui.config(speed=20)
        logging.info(f"Starting the training of max {self.env.maxDataSize} records, for {maxTimeSeconds}s")
        datasetPath = os.path.join(self.rootDir, _DATASET_NAME)
        logger = Logger(datasetPath, self.env.features, self.env.extractor, overwrite)
        gi = self.env.newGame()
        start = time.time()
        while len(logger) <= self.env.maxDataSize and (maxTimeSeconds == 0 or time.time() - start < maxTimeSeconds):
            if gui:
                Gui.render(gi)
                if len(pygame.event.get(eventtype=pygame.QUIT)) > 0:
                    sys.exit()
            action = self.env.trainer.query(gi)
            logger.record(gi, action)
            gameEngine.update(gi, action)
            if gi.isGameOver:
                logging.warning("Game over during training. Recovering...")
                gi = self.env.newGame()
        logger.__del__()
        self.dataset = pandas.read_csv(datasetPath, index_col=0)

    def play(self, player: PlayerI, gui: bool = True, log: bool = False):
        """
        Esegue una partita con un dato giocatore

        Parameters
        ----------
        player: PlayerI
            un player
        gui: bool = True
            abilita o disabilita l'interfaccia grafica
        log: bool = False
            la partita viene utilizzata dal Logger per creare o ampliare un dataset
        """
        if gui:
            Gui.config(speed=30)
        gi = self.env.newGame()
        logger = Logger(os.path.join(self.rootDir, _DATASET_NAME), self.env.features, self.env.extractor) if log else None
        steps = 0
        maxsteps = gi.height * gi.width
        score = 0
        while not gi.isGameOver and steps < maxsteps:
            if gui:
                Gui.render(gi)
                if len(pygame.event.get(eventtype=pygame.QUIT)) > 0:
                    sys.exit()

            action = player.query(gi)
            if logger is not None:
                logger.record(gi, action)
            if gi.score != score:
                steps = 0
                score = gi.score
            else:
                steps += 1
            gameEngine.update(gi, action)
        return gi.score

    def replay(self, first: int = 0, last: int = None):
        """
        Effettua delle partite con il miglior individuo di ogni generazione. Le partite vengono mostrare a video

        Parameters
        ----------
        first: int = 0
            l'indice della history da cui iniziare a effettuare le partire. Ricorda, gli indici partono da 0, il numero della generazione da 1
        last: int = None
            l'indice della history a cui fermarsi (quindi l'individuo all'indice 'last' non viene fatto giocare).
            Se lasciato vuoto, eseguira' fino all'ultimo elemento della history
        """
        history = self.gaInstance.getHistory()
        if last is None:
            last = len(history)
        for i in history[first:last]:
            player = TreeAgent(i[2], self.env.extractor)
            points = self.play(player=player, gui=True)
            print(f"[environment.replay] generation:{i[0]}, fitness:{i[1]}, points={points}")

    def performance(self):
        history = self.gaInstance.getHistory()
        for i in history:
            player = TreeAgent(i[2], self.env.extractor)
            points = 0
            for _ in range(100):
                points += self.play(player=player, gui=False)
            print(f"[environment.replay] generation:{i[0]}, fitness:{i[1]}, avgPoints={points / 100}")

    def printDecisionTree(self, generationNumber: int = None):
        """
        Disegna l'albero decisionale del migliore indivduo della generazione specificata

        Parameters
        ----------
        generationNumber: int
            disegna l'albero della generazione `generationNumber`.
            Se `None` usa l'ultima generazione
        """
        # TODO: "generationNumber" puo' non coincidere con la reale generazione
        # (per evitare un enorme file history e' possibile eliminare righe del file senza comprometterlo)
        generation = self.gaInstance.getHistory()[-1 if generationNumber is None else generationNumber]
        tree = generation[2]
        label = "last" if generationNumber is None else str(generationNumber)
        tree.printToFile(os.path.join(self.rootDir, f"bestTreeOfGen_{label}"))

    def drawHistoryGraph(self, outFile: str = None, accuracy: bool = True):
        """
        Disegna un line chart con l'andamento del fitness nel corso delle generazioni

        Parameters
        ----------
        outFile: str = None
            nome del file su cui salvare il grafico. Se lasciato None, viene usato il nome di default
        accuracy: bool = False
            Inserisce nel grafico anche l'andamento dell'accuratezza degli alberi
        """
        if outFile is None:
            outFile = os.path.join(self.rootDir, self.HISTORY_GRAPH_FILE)
        self.gaInstance.drawHistoryGraph(outFile, accuracy)
