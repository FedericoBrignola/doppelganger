#   SNAKE GAME
#   Author : Apaar Gupta (@apaar97)
#   Python 3.5.2 Pygame
import random
from game.gameInstance import Actions, GameInstance

"""
Il gameEngine si occupa della logica del gioco Snake
"""


def update(gi: GameInstance, action: Actions) -> None:
    """
    Aggiorna un'istanza di gioco con una determinata azione.

    Parameters
    ----------
    gi: GameInstance
        istanza di gioco
    action: Actions
        azione da effettuare
    """

    if gi.isGameOver:
        return

    # Validate direction
    match action:
        case Actions.RIGHT:
            gi.direction = (gi.direction - 1) % 4
        case Actions.LEFT:
            gi.direction = (gi.direction + 1) % 4
            
    # Update snake position
    match gi.direction:
        # Right
        case 0:
            gi.snakePos[0] += 1
        # Up
        case 1:
            gi.snakePos[1] -= 1
        # Left
        case 2:
            gi.snakePos[0] -= 1
        # Down
        case 3:
            gi.snakePos[1] += 1

    # Bounds
    if gi.snakePos[0] >= gi.width or gi.snakePos[0] < 0 or gi.snakePos[1] >= gi.height or gi.snakePos[1] < 0:
        if gi.pacmanWorld:
            gi.snakePos[0] = gi.snakePos[0] % gi.width
            gi.snakePos[1] = gi.snakePos[1] % gi.height
        else:
            gi.isGameOver = True
            
    # Snake body mechanism
    gi.snakeBody.insert(0, list(gi.snakePos))
    if gi.snakePos == gi.foodPos:
        gi.foodSpawn = False
        gi.score += 1
        if not gi.grow:
            gi.snakeBody.pop()
    else:
        gi.snakeBody.pop()
    
    # Food Spawn
    if not gi.foodSpawn:
        while True:
            gi.foodPos = [random.randrange(0, gi.width), random.randrange(0, gi.height)]
            if gi.foodPos not in gi.snakeBody:
                break
        gi.foodSpawn = True

    # Self hit
    for block in gi.snakeBody[1:]:
        if gi.snakePos == block:
            gi.isGameOver = True
