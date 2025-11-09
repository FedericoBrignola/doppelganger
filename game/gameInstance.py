from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


class Actions(Enum):
    """
    Actions contiene le possibili azioni effettuabili nel gioco
    """
    LEFT = auto()
    RIGHT = auto()
    FORWARD = auto()


@dataclass(init=False)
class GameInstance:
    """
    GameInstance mantiene tutte le informazioni relative a una partita di Snake
    """
    foodSpawn: bool
    grow: bool
    pacmanWorld: bool
    width: int
    height: int
    size: Tuple[int, int]

    # Game status
    snakePos: list[int]
    snakeBody: list[list[int]]
    foodPos: list[int]
    direction: int
    score: int
    isGameOver: bool

    def __init__(self) -> None:
        # Constants
        self.foodSpawn: bool = False
        self.grow: bool = False
        self.pacmanWorld: bool = False
        self.width: int = 15
        self.height: int = 10
        self.size: Tuple[int, int] = self.width, self.height

        # Game status
        self.snakePos: list[int] = [3, 5]
        self.snakeBody: list[list[int]] = [[3, 5], [2, 5], [1, 5]]
        self.foodPos: list[int] = [6, 5]
        self.direction: int = 0
        self.score: int = 0
        self.isGameOver: bool = False

    # def __eq__(self, obj: object) -> bool:
    #     return (isinstance(obj, GameInstance)
    #             and self.snakePos == obj.snakePos
    #             and self.snakeBody == obj.snakeBody
    #             and self.foodPos == obj.foodPos
    #             and self.direction == obj.direction
    #             and self.score == obj.score)
    #
    # def __hash__(self) -> int:
    #     return hash((
    #         str(self.snakePos),
    #         str(self.snakeBody),
    #         str(self.foodPos),
    #         self.direction,
    #         self.score
    #     ))
