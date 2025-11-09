import sys
import time

import pygame

from game.gameInstance import Actions, GameInstance
from game.gui import Gui


class Human:
    """
    Agente con albero decisionale

    Methods
    -------
    query(gi: GameInstance) -> Actions:
        interroga l'agente
    """
    _DEFAULT = Actions.FORWARD  # azione di default se l'utente non effettua azioni

    def __init__(self):
        self.actions = list()
        self.last = None
        self.step2step = False

    def query(self, gi: GameInstance) -> Actions:
        """
        Restituisce la prossima mossa del serpente in base agli input di un utente

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco

        Return
        ----------
        Actions: l'azione selezionata dall'utente ("FORWARD" se l'utente non da input)
        """
        Gui.render(gi)
        wait = True
        while len(self.actions) == 0 and wait:
            wait = self.step2step
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.actions.append(0)
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.actions.append(1)
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.actions.append(2)
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.actions.append(3)
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
                    if event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL:
                        self.step2step = not self.step2step

        if len(self.actions) > 0:
            m = self.actions.pop(0)
            # mappatura input "destra/sinistra/su/giu" nelle azioni "vai dritto"(FORWARD), "svolta a destra/sinistra"(RIGHT/LEFT)
            if m == gi.direction or abs(m - gi.direction) == 2:
                return Actions.FORWARD
            elif m == (gi.direction + 1) % 4:
                return Actions.LEFT
            else:
                return Actions.RIGHT
        return self._DEFAULT
