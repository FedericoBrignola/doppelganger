from enum import Enum, auto
from game.gameInstance import Actions, GameInstance


class ZAgent:
    """
    Agente reattivo semplice con movimento a Z

    Methods
    -------
    query(gi: GameInstance) -> Actions:
        interroga l'agente
    """
    @classmethod
    def query(cls, gi: GameInstance) -> Actions:
        """
        Interroga l'agente e restituisce la prossima azione

        Parameters
        ----------
        gi: GameInstance
            Istanza di gioco

        Return
        ------
        Actions: azione da eseguire
        """
        if _foodIsLeft(gi):
            return Actions.LEFT
        if _foodIsRight(gi):
            return Actions.RIGHT
        if _foodIsAhead(gi):
            return Actions.FORWARD
        return _foodBehind(gi)


class LAgent:
    """
    Agente reattivo semplice con movimento a L

    Methods
    -------
    query(gi: GameInstance) -> Actions:
        interroga l'agente
    """
    @classmethod
    def query(cls, gi: GameInstance) -> Actions:
        """
        Interroga l'agente e restituisce la prossima azione

        Parameters
        ----------
        gi: GameInstance
            Istanza di gioco

        Return
        ------
        Actions: azione da eseguire
        """
        if _foodIsAhead(gi):
            return Actions.FORWARD
        if _foodIsLeft(gi):
            return Actions.LEFT
        if _foodIsRight(gi):
            return Actions.RIGHT
        return _foodBehind(gi)


class SAgent:
    """
    Agente reattivo semplice con movimento a S

    Methods
    -------
    query(gi: GameInstance) -> Actions:
        Interroga l'agente
    """
    @classmethod
    def query(cls, gi: GameInstance) -> Actions:
        """
        Interroga l'agente e restituisce la prossima azione

        Parameters
        ----------
        gi: GameInstance
            Istanza di gioco

        Return
        ------
        Actions: azione da eseguire
        """
        # se il serpente sta andando verso l'alto vuol dire che sta tornando in cima al campo
        if gi.direction == 1:
            # Se puo' ancora salire, continua
            if gi.snakePos[1]-1 >= 0:
                return Actions.FORWARD
            # altrimenti, se e' salito costeggiando la parte sinistra del campo, svolta a destra
            if gi.snakePos[0] == 0:
                return Actions.RIGHT
            # altrimenti è salito sulla parte destra del campo quindi svolta a sinistra
            return Actions.LEFT
        
        # se sta andando a destra
        if gi.direction == 0:
            # Continua finche' non arriva al bordo del campo
            if gi.snakePos[0]+1 < gi.width:
                return Actions.FORWARD
            # se e' arrivato al bordo del campo e c'e' una casella sotto, svolta a destra
            if gi.snakePos[1]+1 < gi.height:
                return Actions.RIGHT
            # se non ci sono caselle sotto, svolta a sinistra
            return Actions.LEFT
        
        # se sta andando a sinistra
        if gi.direction == 2:
            # Continua finche' non arriva al bordo del campo
            if gi.snakePos[0]-1 >= 0:
                return Actions.FORWARD
            # se e' arrivato al bordo del campo e c'e' una casella sotto, svolta a sinistra
            if gi.snakePos[1]+1 < gi.height:
                return Actions.LEFT
            # se non ci sono caselle sotto, svolta a destra
            return Actions.RIGHT

        # Se sta andando verso il basso
        if gi.direction == 3:
            # svolta a sinistra se si trova sul lato sinistro del campo
            if gi.snakePos[0] == 0:
                return Actions.LEFT
            # Altrimenti a destra
            return Actions.RIGHT
        
        return Actions.FORWARD


def _foodBehind(gi: GameInstance):
    # Se il serpente va dalla parte opposta al cibo e puo' girare a sinistra, allora gira a sinistra, altrimenti a destra
    if (
        (gi.direction == 0 and gi.snakePos[1] >= gi.foodPos[1] and gi.snakePos[1] > 0) or            # serpente con cibo a sinistra e che va verso destra
        (gi.direction == 1 and gi.snakePos[0] >= gi.foodPos[0] and gi.snakePos[0] > 0) or            # serpente con cibo in basso e che va verso l'alto
        (gi.direction == 2 and gi.snakePos[1] <= gi.foodPos[1] and gi.snakePos[1]+1 < gi.height) or  # serpente con cibo a destra e che va verso sinistra
        (gi.direction == 3 and gi.snakePos[0] <= gi.foodPos[0] and gi.snakePos[0]+1 < gi.width)      # serpente con cibo in alto e che va verso il basso
    ):
        return Actions.LEFT
    return Actions.RIGHT


def _foodIsAhead(gi: GameInstance):
    # True se il serpente va verso il cibo
    return (
        (gi.snakePos[0] < gi.foodPos[0] and gi.direction == 0) or  # serpente con cibo a destra e che va verso destra
        (gi.snakePos[1] > gi.foodPos[1] and gi.direction == 1) or  # serpente con cibo in alto e che va verso l'alto
        (gi.snakePos[0] > gi.foodPos[0] and gi.direction == 2) or  # serpente con cibo a sinistra e che va verso sinistra
        (gi.snakePos[1] < gi.foodPos[1] and gi.direction == 3)     # serpente con cibo in basso e che va verso il basso
    )


def _foodIsLeft(gi: GameInstance):
    # True se il serpente ha il cibo sulla sinistra
    return (
        (gi.snakePos[1] > gi.foodPos[1] and gi.direction == 0) or  # Serpente con cibo in alto e che va verso destra
        (gi.snakePos[0] > gi.foodPos[0] and gi.direction == 1) or  # Serpente con cibo a sinistra e che va verso l'alto
        (gi.snakePos[1] < gi.foodPos[1] and gi.direction == 2) or  # Serpente con cibo in basso e che va verso sinistra
        (gi.snakePos[0] < gi.foodPos[0] and gi.direction == 3)     # Serpente con cibo a destra e che va verso il basso
    )


def _foodIsRight(gi: GameInstance):
    # True se il serpente ha il cibo sulla destra
    return (
        (gi.snakePos[0] > gi.foodPos[0] and gi.direction == 3) or  # Serpente a sinistra del cibo e che va verso il basso
        (gi.snakePos[0] < gi.foodPos[0] and gi.direction == 1) or  # Serpente a destra del cibo e che va verso l'alto
        (gi.snakePos[1] > gi.foodPos[1] and gi.direction == 2) or  # Serpente sotto il cibo e che va verso sinistra
        (gi.snakePos[1] < gi.foodPos[1] and gi.direction == 0)     # Serpente sopra il cibo e che va verso destra
    )
