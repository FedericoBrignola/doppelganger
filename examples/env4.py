import sys,os
"""
[ IMPORTANT ]   the project should be an installable python module,
                for now, we manually inject the module path into "sys.path"
"""
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/..")

import pandas as pd
from sklearn.model_selection import train_test_split

from environment import Environment, Console
from game.gameInstance import GameInstance
from player.simpleReactiveAgent import LAgent
from training import geneticAlgorithm
from training.decisionTree3 import FeatureSet

import training.algorithms as alg

env = Environment()
gi = GameInstance()
gi.grow = False

features = FeatureSet()
domain = ['w', 'f', 'b', 'h']
"""
    w = cella bianca
    f = cibo
    b = corpo serpente
    h = testa serpente
"""
for x in range(gi.width):
    for y in range(gi.height):
        # features.addNew(f"[{x}:{y}]", values=domain)
        features.add(f"[{x}:{y}]", False)


def fExtractor(g: GameInstance):

    values = dict()
    for xi in range(g.width):
        for yi in range(g.height):
            values[f"[{xi}:{yi}]"] = 'w'
    values[f"[{g.foodPos[0]}:{g.foodPos[1]}]"] = 'f'
    for xi, yi in g.snakeBody:
        values[f"[{xi}:{yi}]"] = 'b'
    values[f"[{g.snakePos[0]}:{g.snakePos[1]}]"] = 'h'

    return values


def mutation(*args):
    pass


def stop(*args):
    return False


def algorithm(dataset: pd.DataFrame):
    trainingSet, testingSet = train_test_split(dataset, test_size=0.5, shuffle=True)

    # noinspection PyTypeChecker
    return geneticAlgorithm.Pack(
        populationGenerator=alg.firstGeneration.Id3Generation(trainingSet, features, 4),
        fitness=alg.fitness.WeightedFormula(1, 0.8, testingSet, trainingSet),
        selection=alg.selection.WheelSelection(),
        crossover=alg.crossover.SemanticSubtreeCrossoverSameFeature(dataset),
        mutation=mutation,
        stopCondition=stop,
        draws=30,
        elitismSize=3
    )


env.setTrainer(trainer=LAgent(), maxDataSize=30000)
env.setAlgorithm(algorithm)
env.setFeatures(features, fExtractor)
env.setGameEnv(gi)

if __name__ == "__main__":
    c = Console("local/env4/", env)
    # c.train(maxTimeSeconds=10)
    # c.runAlgorithm()
    c.replay()
    # c.drawHistoryGraph()
    # c.printDecisionTree()
    # c.performance()