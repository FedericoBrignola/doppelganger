class NGenerationStop:
    """
    L'algoritmo termina dopo nGen generazioni
    """
    def __init__(self, nGen: int):
        """
        Parameters
        ----------
        nGen: int
            numero di generazione a cui interrompere l'esecuzione
        """
        self.nGen = nGen

    def __call__(self, generationN: int, bestFitness: float, avgFitness: float) -> bool:
        return generationN >= self.nGen


class BestFitnessStop:
    """
    L'algoritmo termina quando l' individuo con fitness piu' alto della generazione raggiunge un certo valore
    """
    def __init__(self, fitness: float):
        """
        Parameters
        ----------
        fitness: float
            la soglia di fitness da raggiungere
        """
        self.fitness = fitness

    def __call__(self, generationN: int, bestFitness: float, avgFitness: float) -> bool:
        return bestFitness >= self.fitness
