from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Protocol

import pandas

if TYPE_CHECKING:
    from multiprocessing.pool import ApplyResult
    from typing import TextIO

import multiprocessing
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from training.decisionTree3 import DecisionTree


# Static types
class TGenPopulation(Protocol):
    """
    Classe generica per generare la popolazione iniziale
    """

    def __call__(self) -> list[DecisionTree]: ...


class TFitness(Protocol):
    """
    Classe generica per calcolare il fitness di un albero decisionale
    """

    def __call__(self, elem: DecisionTree) -> float: ...


class TSelection(Protocol):
    """
    Classe generica per la funzione di selezione
    """

    def __call__(self, population: list[DecisionTree], fitness: list[float]) -> list[DecisionTree]: ...


class TCrossover(Protocol):
    """
    Classe generica per la funzione di crossover
    """

    def __call__(self, parents: list[DecisionTree]) -> list[DecisionTree]: ...


class TMutation(Protocol):
    """
    Classe generica per la funzione di mutazione
    """

    def __call__(self, elem: DecisionTree) -> None: ...


class TStopCondition(Protocol):
    """
    Classe generica per la condizione di stop
    """

    def __call__(self, generationN: int, bestFitness: float, avgFitness: float) -> bool: ...


# Rappresenta un algoritmo genetico "pronto all'uso"
# utilizzato per configurare i parametri e funzioni dell'algoritmo senza doverlo inizializzare
GeneticAlgorithm = Callable[['GA'], None]
HistoryRow = tuple[int, float, DecisionTree, float]
History = list[HistoryRow]


class Pack:
    """
    Configurazione di un algoritmo genetico
    """

    def __init__(self,
                 populationGenerator: TGenPopulation,   # Funzione per generare la popolazione iniziale
                 fitness: TFitness,                     # Funzione di fitness
                 selection: TSelection,                 # Funzione di selezione
                 crossover: TCrossover,                 # Funzione di crossover
                 mutation: TMutation,                   # Funzione di mutazione
                 stopCondition: TStopCondition,         # Criterio di stop
                 draws: int = 25,                       # Numero di iterazioni del crossover per ogni generazione
                 elitismSize: int = 0):                 # Numero di elementi della popolazione corrente da mantenere nella nuova popolazione
        """
        Costruttore Pack

        Parameters
        ----------
        populationGenerator: TGenPopulation
            Funzione per generare la popolazione iniziale
        fitness: TFitness
            Funzione di fitness
        selection: TSelection
            Funzione di selezione
        crossover: TCrossover
            Funzione di crossover
        mutation: TMutation
            Funzione di mutazione
        stopCondition: TStopCondition
            Criterio di stop
        draws: int = 25
            numero di iterazioni del crossover per ogni generazione
            questo definisce la grandezza della popolazione
            per esempio: con un crossover che genera 2 nuovi figli,
            un draws di 25 significa una popolazione di 50 individui
        elitismSize: int = 0
            Se diverso da 0, i migliori n individui della generazione corrente
            (con n = elitismSize) verranno inseriti nella prossima generazione
            senza essere modificati
        """
        self.stopCondition = stopCondition
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.fitness = fitness
        self.genPopulation = populationGenerator
        self.draws = draws
        self.elitismSize = elitismSize


class GA:
    """
    Questa classe rappresenta lo scheletro di un generico algoritmo genetico.

    Methods
    -------
    loadHistory(historyPath: str) -> History
        carica una history precedentemente salvata su file e la restituisce
    saveHistoryRow(file: TextIO, data: HistoryRow) -> None
        Salva su file una riga della history
    run() -> None
        esegue il GA con le configurazioni date
    """
    # TODO: history[g] = (f,i) / array contenente i migliori individui (fitness, individuo) per ogni generazione g

    HISTORY_FILE_NAME = "history.ga"
    FIRST_GEN_LOCATION = "firstgen"
    LAST_GEN_LOCATION = "lastgen"
    TESTING_SET_NAME = "testing.csv"

    def __init__(self, pack: Pack, datastorePath: str):
        """
        Parameters
        ----------
        pack: Pack
            oggetto Pack contenente le configurazioni del GA
        datastorePath: str
            path in cui verranno salvati tutti i dati relativi all'esecuzione del GA
        """
        # Algorithm configurations
        self.stopCondition: TStopCondition = pack.stopCondition
        self.mutation: TMutation = pack.mutation
        self.crossover: TCrossover = pack.crossover
        self.selection: TSelection = pack.selection
        self.fitness: TFitness = pack.fitness
        self.genPopulation: TGenPopulation = pack.genPopulation
        self.draws: int = pack.draws
        self.elitismSize: int = pack.elitismSize

        # Internal initializations
        self.datastorePath: str = datastorePath
        self.population: list[DecisionTree] = list()

        # Prova a caricare in 'self.population' ogni .dtree nella directory GA.LAST_GEN_LOCATION nel datastore;
        # se non presente, prova a caricare i .dtree nella directory GA.FIRST_GEN_LOCATION nel datastore;
        for loc in [GA.LAST_GEN_LOCATION, GA.FIRST_GEN_LOCATION]:
            path = os.path.join(self.datastorePath, loc)
            if os.path.isdir(path):
                if len(self.population) == 0:
                    for f in os.listdir(path):
                        if f.endswith(".dtree"):
                            self.population.append(DecisionTree.load(os.path.join(path, f)))
            else:
                os.mkdir(path)

    def getHistory(self) -> History:
        """History e' una lista di tuple della forma [<generation number>, <highest fitness>, <DecisionTree with highest fitness>, <avg Fitness>]"""
        return GA._loadHistory(os.path.join(self.datastorePath, GA.HISTORY_FILE_NAME))


    @staticmethod
    def _loadHistory(historyPath: str) -> History:
        """
        Carica una history da file

        Parameters
        ----------
        historyPath: str
            path del file contenente la history da caricare

        Return
        ------
        History: la history dove ogni elemento della lista e' della forma [numGenerazione, bestFitness, DecisionTreeBestFitness, avgFitness]
        """
        data = list()
        if os.path.isfile(historyPath):
            with open(historyPath, "r") as file:
                line = file.readline()
                while line:
                    data.append(pickle.loads(bytes.fromhex(line)))
                    line = file.readline()
        return data

    @staticmethod
    def saveHistoryRow(file: TextIO, data: HistoryRow) -> None:
        """
        Salva su file una riga di history

        Parameters
        ----------
        file: todo
        data: HistoryRow
            i dati da caricare nel file. Sono della forma [numGenerazione, bestFitness, DecisionTreeBestFitness, avgFitness]
        """
        file.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL).hex() + "\n")

    def getAccuracy(self):
        print("[GA] - start accuracy")
        testingSet = pandas.read_csv(os.path.join(self.datastorePath, GA.TESTING_SET_NAME), index_col=0)
        history = self.getHistory()
        accuracyVector = []
        for g in history:
            tree = g[2]
            correctPredictions = 0
            for i, row in testingSet.iterrows():
                if tree.predict(row.to_dict()) == i:
                    correctPredictions += 1
            accuracy = correctPredictions / testingSet.index.size
            accuracyVector.append(accuracy)
            print(f"[GA] - best tree accuracy gen {g[0]}: {round(accuracy*100, 2)}")
        return accuracyVector

    def drawHistoryGraph(self, outFile: str, accuracy: bool) -> None:
        """
        Disegna un line chart del fitness delle varie generazioni generate

        Parameters
        ----------
        outFile: str
            file su cui salvare il grafico generato
        accuracy: bool
            Inserisce nel grafico anche l'andamento dell'accuratezza degli alberi
        """
        history = self.getHistory()
        xdata = []
        bestFitness = []
        avgFitness = []
        for g, mF, _, avgF in history:
            xdata.append(g)
            bestFitness.append(mF)
            avgFitness.append(avgF)
        fig, ax = plt.subplots()
        ax.plot(xdata, bestFitness, label="bestFitness")
        ax.plot(xdata, avgFitness, label="avgFitness")
        if accuracy:
            ax.plot(xdata, self.getAccuracy(), label="accuracy")
        ax.legend()
        ax.set_xlabel("generation")
        ax.set_ylabel("niceness")
        ax.grid(linestyle="--", color="#d8e1ed")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0, y=0.05)
        plt.savefig(outFile + ".png")

    @staticmethod
    def _worker(selectionFunction, crossoverFunction, mutationFunction, population, fitnessPopulation):
        """
        Funzione usata per la multiprogrammazione
        """
        parents = selectionFunction(population, fitnessPopulation)
        newElements = crossoverFunction(parents)
        for e in newElements:
            mutationFunction(e)
        return newElements

    def run(self, processes: int | None = None):
        """
        Esecuzione dell'algoritmo genetico con le configurazioni date
        """
        # Try to resume a previously computed generation
        history = self.getHistory()
        bestFitness = avgFitness = generation = 0
        if len(history) > 0:
            generation = history[-1][0]
            bestFitness = history[-1][1]
            avgFitness = history[-1][3]
            print(f"[GA.run] Resuming GA execution: generation:{generation}  best_fitness:{bestFitness}  avgFitness:{avgFitness}")
        else:
            print(f"[GA.run] Starting GA execution")

        fitnessPopulation: list[float] = list()
        results: list[ApplyResult] = list()  # usato per la multiprogrammazione
        # se e' la prima generazione, usa la funzione genPopulation e salva su file gli alberi generati
        if len(self.population) == 0:
            self.population = self.genPopulation()
            for i, tree in enumerate(self.population):
                tree.save(os.path.join(self.datastorePath, GA.FIRST_GEN_LOCATION, str(i)), persistent=False)
        with open(os.path.join(self.datastorePath, GA.HISTORY_FILE_NAME), 'a') as archive, multiprocessing.Pool(processes=processes) as pool:
            while (self.stopCondition is None
                   or not self.stopCondition(generation, bestFitness, avgFitness)):

                generation += 1
                fitnessPopulation.clear()
                results.clear()
                for elem in self.population:
                    results.append(pool.apply_async(self.fitness, [elem]))  # Calcolo del fitness degli alberi con la multiprogrammazione
                for result in results:
                    fitnessPopulation.append(result.get())

                newPopulation = list()
                results.clear()
                # draws e' il numero di iterazioni del crossover per ogni generazione
                # per ogni draws, lancia quindi la funzione _worker in parallelo ottenendo gli individui della nuova generazione
                for _ in range(self.draws):
                    results.append(pool.apply_async(GA._worker, [self.selection, self.crossover, self.mutation, self.population, fitnessPopulation]))
                for r in results:
                    newPopulation.extend(r.get())

                # se e' stato impostato l'elistmSize, aggiunge gli individui migliore della generazione corrente nella prossima generazione
                if self.elitismSize > 0:
                    if len(fitnessPopulation) < self.elitismSize:
                        newPopulation.extend(self.population)
                    else:
                        elitism = np.argpartition(fitnessPopulation, -self.elitismSize)[-self.elitismSize:]  # ottiene gli indici degli individui con miglior fitness
                        for index in elitism:
                            newPopulation.append(self.population[index])

                # HISTORY management
                # Salva nella history i dati relativi alla generazione corrente
                bestFitnessIndex = np.argmax(fitnessPopulation)  # ottiene l'indice dell'individuo con miglior fitness
                bestFitness = fitnessPopulation[bestFitnessIndex]
                avgFitness = sum(fitnessPopulation) / len(fitnessPopulation)  # calcola la media del fitness della popolazione
                GA.saveHistoryRow(archive, (generation, bestFitness, self.population[bestFitnessIndex], avgFitness))
                print(f"[GA.run] generation:{generation}  best_fitness:{bestFitness}  avgFitness:{avgFitness}")
                # Salva su file la nuova popolazione sovrascrivendo la precedente
                for n, i in enumerate(newPopulation):
                    i.save(f"{self.datastorePath}/{GA.LAST_GEN_LOCATION}/{n}", persistent=False)

                self.population = newPopulation
        print(f"[GA.run] Terminated")
