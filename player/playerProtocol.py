from typing import Protocol

from game.gameInstance import GameInstance, Actions


# TODO: Rename
class PlayerI(Protocol):
    """
    Classe usata come duck typing

    Methods
    -------
    query(gi: GameInstance) -> Actions
        Interroga il player per l'azione successiva
    """
    def query(self, gi: GameInstance) -> Actions: ...
