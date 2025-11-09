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
from player import humanAgent
from player.humanAgent import Human
from player.simpleReactiveAgent import LAgent, SAgent, ZAgent
from training import geneticAlgorithm
from training.decisionTree3 import FeatureSet

import training.algorithms as alg


env = Environment()
gi = GameInstance()
gi.grow = True
gi.pacmanWorld = True


features = FeatureSet()
features.add("snake[0]", True)
features.add("snake[1]", True)
features.add("food[0]", True)
features.add("food[1]", True)
features.add("direction", False)


def fExtractor(g: GameInstance):
    return features.setValues({'snake[0]': g.snakePos[0],
                               'snake[1]': g.snakePos[1],
                               'food[0]': g.foodPos[0],
                               'food[1]': g.foodPos[1],
                               'direction': g.direction})


def mutation(*args):
    pass


def stop(*args):
    return False


def algorithm(dataset: pd.DataFrame):
    trainingSet, testingSet = train_test_split(dataset, test_size=0.5, shuffle=True)

    # noinspection PyTypeChecker
    return geneticAlgorithm.Pack(
        populationGenerator=alg.firstGeneration.Id3Generation(trainingSet, features, 5),
        fitness=alg.fitness.WeightedFormula4(1, 1, testingSet, trainingSet),
        selection=alg.selection.WheelSelection(),
        crossover=alg.crossover.SemanticSubtreeCrossoverSameFeature(dataset),
        mutation=mutation,
        stopCondition=stop,
        draws=30,
        elitismSize=10
    )


env.setTrainer(trainer=Human(), maxDataSize=30000)
env.setAlgorithm(algorithm)
env.setFeatures(features, fExtractor)
env.setGameEnv(gi)

if __name__ == "__main__":
    c = Console("local/env5/", env)
    c.train(maxTimeSeconds=1000, overwrite=False)
    # c.runAlgorithm()
    # c.replay()
