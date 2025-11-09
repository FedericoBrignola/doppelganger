from __future__ import annotations

from game.gameInstance import Actions, GameInstance
from training.decisionTree3 import DecisionTree, FeatureSet


class TreeAgent:
    """
    Agente con albero decisionale

    Methods
    -------
    load(agentFile: str) -> TreeAgent
        Permette il caricamento di un albero decisionale a partire da un gile
    query(gi: GameInstance) -> Actions:
        interroga l'agente
    """

    def __init__(self, tree: DecisionTree, featureExtractor: callable[[GameInstance], dict[str, any]]) -> None:
        self.tree = tree
        self.extractor = featureExtractor

    @classmethod
    def load(cls, agentfile: str) -> TreeAgent:
        """
        Permette il caricamento dell'albero a partire da un file

        Parameters
        ----------
        agentfile: str
            path file albero

        Return
        ------
        TreeAgent: istanza di TreeAgent
        """
        return TreeAgent(DecisionTree.load(agentfile))

    def query(self, gi: GameInstance) -> Actions:
        """
        Interroga l'albero decisionale e restituisce la prossima mossa

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco

        Return
        ------
        Actions: azione da eseguire
        """
        features = self.extractor(gi)
        action = self.tree.predict(features)
        if action is None:
            print(f"[ WARN ] null prediction with instance: {features}")
            return Actions.FORWARD
        return Actions[action]
