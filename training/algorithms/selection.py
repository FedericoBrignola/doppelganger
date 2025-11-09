import numpy as np

from training.decisionTree3 import DecisionTree


class WheelSelection:
    """
    Wheel Selection. Le probabilita' di un individuo di essere selezionato, sono pari al proprioFitness / totalFitness
    """
    @staticmethod
    def __call__(population: list[DecisionTree], fitness: list[float]) -> list[DecisionTree, DecisionTree]:
        sumFitness = sum(fitness)
        s = np.random.default_rng().choice(population, 2, replace=len(population) < 2, p=[x / sumFitness for x in fitness])
        return s


class TournamentSelection:
    """
    Tournament selection
    """
    def __init__(self, tournamentSize: int, prob: float, winnersSize: int = 2):
        """
        Parameters
        ----------
        tournamentSize: int
            e' il numero d'individui che vengono selezionati per partecipare a un torneo
        prob: float
            valore tra [0,1]. E' la probabilita' con cui verranno selezionati gli individui
        winnersSize: int = 2
            il numero di vincitori al termine di un torneo
        """
        self.tournamentSize = tournamentSize
        self.winnersSize = winnersSize
        self.prob = prob

    def __call__(self, population: list[DecisionTree], fitness: list[float]) -> list[DecisionTree, DecisionTree]:
        winners = []
        zipped = list(zip(population, fitness))
        r = np.random.default_rng()
        for _ in range(self.winnersSize):
            currentTournamentPopulation = list(r.choice(zipped, self.tournamentSize, replace=self.tournamentSize > len(zipped)))  # seleziona randomicamente tournamentSize individui senza rimpiazzo
            currentTournamentPopulation.sort(key=lambda el : el[1], reverse=True)             # ordina gli individui selezionati in base al fitness decrescente
            i = 0
            while True:
                el = currentTournamentPopulation[i % self.tournamentSize]
                if r.random() < self.prob:
                    winner = el
                    break
                i += 1
            winners.append(winner[0])
        return winners
