# import sys,os
# """
# [ IMPORTANT ]   the project should be an installable python module,
#                 for now, we manually inject the module path into "sys.path"
# """
# sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/..")
import pathlib

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
features.addNew("snake[0]", minValue=0, maxValue=14)
features.addNew("snake[1]", minValue=0, maxValue=9)
features.addNew("food[0]", minValue=0, maxValue=14)
features.addNew("food[1]", minValue=0, maxValue=9)
features.addNew("direction", values=["1", "2", "3", "4"])


def fExtractor(g: GameInstance):
    return {'snake[0]': g.snakePos[0],
            'snake[1]': g.snakePos[1],
            'food[0]': g.foodPos[0],
            'food[1]': g.foodPos[1],
            'direction': g.direction}


def algorithm(dataset: pd.DataFrame):
    trainingSet, testingSet = train_test_split(dataset, test_size=0.8, shuffle=True)

    # noinspection PyTypeChecker
    return geneticAlgorithm.Pack(
        populationGenerator=alg.firstGeneration.Id3Generation(trainingSet, features, 10),
        fitness=alg.fitness.AccuracyBasedFitnessPwm2(dataset),
        selection=alg.selection.WheelSelection(),
        crossover=alg.crossover.SemanticSubtreeCrossoverSameFeature(dataset),
        mutation=alg.mutation.ChangeTestMutation(),
        stopCondition=alg.stop.NGenerationStop(30),
        draws=27,
        elitismSize=6
    )


env.setTrainer(trainer=LAgent(), maxDataSize=10000)
env.setAlgorithm(algorithm)
env.setFeatures(features, fExtractor)
env.setGameEnv(gi)

if __name__ == "__main__":
    c = Console(__file__, env)
    # c.train()
    c.runAlgorithm()
    c.drawHistoryGraph(accuracy=True)
    c.printDecisionTree()
    # c.replay()