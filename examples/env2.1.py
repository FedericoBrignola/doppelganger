import pandas as pd
from sklearn.model_selection import train_test_split

from environment import Environment, Console
from game.gameInstance import GameInstance
from player import humanAgent
from player.simpleReactiveAgent import LAgent
from training import geneticAlgorithm
from training.decisionTree3 import FeatureSet

import training.algorithms as alg

env = Environment()
gi = GameInstance()
gi.grow = False

features = FeatureSet()
features.add("foodDir", False) # Food direction Relative To Snake {F,R,L,++,--,+-,-+}
features.add("wallN", False)
features.add("wallE", False)
features.add("wallS", False)
features.add("wallW", False)


def fExtractor(g: GameInstance):
    """
    L'origine delle assi viene traslata in `snakePos` e il piano rotato in relazione alla direzione del serpente
    Rotazione:
        se il serpente va verso Nord(1) non viene ruotato
        se verso Est(0) rotazione di -90
        se verso Ovest(2) rotazione di 90
        se verso Sud(3) rotazione di 180
    """
    # axis translation and rotation
    match g.direction:
        case 2:
            # rotation of 90 degree
            relativeFposX = -(g.foodPos[1] - g.snakePos[1])
            relativeFposY = g.foodPos[0] - g.snakePos[0]
        case 0:
            # rotation of -90 degree
            relativeFposX = g.foodPos[1] - g.snakePos[1]
            relativeFposY = -(g.foodPos[0] - g.snakePos[0])
        case 3:
            # rotation of 180 degree
            relativeFposX = -(g.foodPos[0] - g.snakePos[0])
            relativeFposY = -(g.foodPos[1] - g.snakePos[1])
        case _:
            # only translation
            relativeFposX = g.foodPos[0] - g.snakePos[0]
            relativeFposY = g.foodPos[1] - g.snakePos[1]

    if relativeFposX > 0:
        if relativeFposY == 0:
            foodDir = "R"
        elif relativeFposY > 0:
            foodDir = "++"
        else:
            foodDir = "+-"
    elif relativeFposX < 0:
        if relativeFposY == 0:
            foodDir = "L"
        elif relativeFposY > 0:
            foodDir = "-+"
        else:
            foodDir = "--"
    else:
        if relativeFposY > 0:
            foodDir = "F"
        else:
            foodDir = "B"

    test = {'foodDir': foodDir,
            'wallN': g.snakePos[1] == 0,
            'wallE': g.snakePos[0] == g.width,
            'wallS': g.snakePos[1] == g.height,
            'wallW': g.snakePos[0] == 0,
            }
    return test


def mutation(*args):
    pass


def algorithm(dataset: pd.DataFrame):
    trainingSet, testingSet = train_test_split(dataset, test_size=0.8, shuffle=True)

    # noinspection PyTypeChecker
    return geneticAlgorithm.Pack(
        populationGenerator=alg.firstGeneration.Id3Generation(trainingSet, features, 10),
        fitness=alg.fitness.WeightedFormula(1, 0.8, testingSet, trainingSet),
        selection=alg.selection.WheelSelection(),
        crossover=alg.crossover.SemanticSubtreeCrossoverSameFeature(dataset),
        mutation=mutation,
        stopCondition=alg.stop.NGenerationStop(30),
        draws=27,
        elitismSize=6
    )


env.setTrainer(trainer=LAgent(), maxDataSize=30000)
env.setAlgorithm(algorithm)
env.setFeatures(features, fExtractor)
env.setGameEnv(gi)

if __name__ == "__main__":
    c = Console("local/env2.1/", env)
    c.train(maxTimeSeconds=10)
    c.runAlgorithm()
    c.drawHistoryGraph()
    # c.printDecisionTree()
    # c.replay()
    c.performance()
