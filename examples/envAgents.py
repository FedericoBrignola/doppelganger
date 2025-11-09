import sys,os

from player import humanAgent

"""
[ IMPORTANT ]   the project should be an installable python module,
                for now, we manually inject the module path into "sys.path"
"""
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/..")

import pandas as pd
from sklearn.model_selection import train_test_split
from environment import Environment, Console
from game.gameInstance import GameInstance
from player.simpleReactiveAgent import LAgent, SAgent, ZAgent
from training import geneticAlgorithm
from training.decisionTree3 import FeatureSet

import training.algorithms as alg

env = Environment()
gi = GameInstance()
gi.width = 40
gi.height = 30
gi.size = gi.width, gi.height
gi.pacmanWorld = True
gi.grow = False

features = FeatureSet()
features.add("snake[0]", True)
features.add("snake[1]", True)
features.add("food[0]", True)
features.add("food[1]", True)
features.add("direction", False)


def fExtractor(g: GameInstance):
    return {'snake[0]': g.snakePos[0],
            'snake[1]': g.snakePos[1],
            'food[0]': g.foodPos[0],
            'food[1]': g.foodPos[1],
            'direction': g.direction}


env.setTrainer(trainer=ZAgent(), maxDataSize=30000)
env.setAlgorithm(lambda d: geneticAlgorithm.Pack(None,None,None,None,None,None))
env.setFeatures(features, fExtractor)
env.setGameEnv(gi)

if __name__ == "__main__":
    c = Console("local/envHuman/", env)
    c.train(gui=True)
    # c.runAlgorithm()
    # c.drawHistoryGraph()
    # c.printDecisionTree()
    # c.replay()

