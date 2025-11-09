import sys
import pygame

from game.gameInstance import GameInstance

# Colors
red = pygame.Color(237, 85, 59)
snakeBodyColor = pygame.Color(88, 24, 69)
snakeHeadColor = pygame.Color(199, 0, 57)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
foodColor = pygame.Color(255, 195, 0)


class Gui:
    delta = 16
    fps = 30
    _playSurface = None
    _fpsController = None

    @classmethod
    def config(cls, speed: int = None, delta: int = None) -> None:
        """
        Configura la finestra di gioco.

        Parameters
        ----------
        speed: int = None
            velocita' del serpente in passi per secondo
        delta: int = None
            grandezza di un quadrato di gioco espressa in pixels
        """
        if speed is not None:
            cls.fps = speed
        if delta is not None:
            cls.delta = delta

    @classmethod
    def _pygameInit(cls, gi: GameInstance):
        """
        Inizializza la finestra di gioco.

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco
        """
        # Pygame Init
        init_status = pygame.init()
        if init_status[1] > 0:
            print("(!) Had {0} initialising errors, exiting... ".format(init_status[1]))
            sys.exit()
        pygame.display.set_caption("Snake Game")
        cls._playSurface = pygame.display.set_mode((gi.width * cls.delta, gi.height * cls.delta))
        cls._fpsController = pygame.time.Clock()


    @classmethod
    def __gameOver(cls, gi: GameInstance) -> None:
        """
        Finestra di game over con il punteggio raggiunto

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco
        """
        myFont = pygame.font.SysFont('monaco', 64)
        GOsurf = myFont.render("Game Over", True, red)
        GOrect = GOsurf.get_rect()
        GOrect.midtop = (int(gi.width/2)*cls.delta, int(gi.height / 3) * cls.delta)
        cls._playSurface.blit(GOsurf, GOrect)
        cls.__showScore(gi)
        pygame.display.flip()
        # time.sleep(4)
        # pygame.quit()
        # sys.exit()


    # Show Score
    @classmethod
    def __showScore(cls, gi: GameInstance, choice=0) -> None:
        """
        Finestra che mostra lo score

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco
        choice: int =0
            se lo score e' da mostrare nella finestra di gameOver, lascia `choice=0`, se invece va mostrato in una finestra a parte, usa `choice=1`
        """

        SFont = pygame.font.SysFont('monaco', 32)
        Ssurf = SFont.render("Score  :  {0}".format(gi.score), True, black)
        Srect = Ssurf.get_rect()
        if choice == 1:
            Srect.midtop = (int(gi.width / 4) * cls.delta, int(gi.height / 3) * cls.delta)
        else:
            Srect.midtop = (int(gi.width / 2) * cls.delta, 25)
        cls._playSurface.blit(Ssurf, Srect)


    @classmethod
    def render(cls, gi: GameInstance) -> None:
        """
        Mostra un'istanza di gioco a video

        Parameters
        ----------
        gi: GameInstance
            istanza di gioco
        """
        delta = cls.delta
        if not pygame.get_init():
            cls._pygameInit(gi)

        pygame.event.pump() # pulisce gli eventi. Evita messaggio "la finestra non risponde"
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()

        if gi.isGameOver:
            # self.__gameOver(gi)
            return

        cls._playSurface.fill(white)
        # Food
        pygame.draw.rect(cls._playSurface, foodColor, pygame.Rect(gi.foodPos[0] * delta, gi.foodPos[1] * delta, delta, delta))
        # Snake body
        for pos in gi.snakeBody[1:]:
            pygame.draw.rect(cls._playSurface, snakeBodyColor, pygame.Rect(pos[0] * delta, pos[1] * delta, delta, delta))
        # Snake head
        pygame.draw.rect(cls._playSurface, snakeHeadColor, pygame.Rect(gi.snakePos[0] * delta, gi.snakePos[1] * delta, delta, delta))
        pygame.display.flip()
        cls._fpsController.tick(cls.fps)
