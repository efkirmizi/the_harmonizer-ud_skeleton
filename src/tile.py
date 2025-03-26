import pygame
import src.settings as settings

"""
#############################################
    DO NOT CHANGE ANYTHING IN THIS FILE.
#############################################
"""


class Tile(pygame.sprite.Sprite):
    def __init__(self, game, x, y, text):
        """
            Initializes the tile object class.

            Args:
                game (Game): Game object
                x (int): X coordinate of the tile
                y (int): Y coordinate of the tile
                text (str): Text to be displayed on the tile
        """
        pygame.font.init()
        self.groups = game.all_sprites
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.x, self.y = x, y
        self.text = text

        self.image = pygame.Surface((settings.TILESIZE, settings.TILESIZE))
        self.rect = self.image.get_rect()
        
        # Set the corresponding background image to the tile.
        if self.text == "0":
            self.image.fill(settings.WHITE)
        elif self.text == "1":
            self.image.fill(settings.RED)
        elif self.text == "2":
            self.image.fill(settings.BLUE)
        elif self.text == "3":
            self.image.fill(settings.PURPLE)
        elif self.text == "4":
            self.image.fill(settings.BLACK)
        elif self.text in ["5", "6", "7", "8"]:
            # Tile center
            tc = [settings.TILESIZE // 2, settings.TILESIZE // 2]
            self.image.fill(settings.YELLOW)
            conveyor_id = int(self.text) - 5
            # Arrow direction
            ad = [conveyor_id // 2 * (1 - 2 * (conveyor_id % 2)), (1 - conveyor_id // 2) * (1 - 2 * (conveyor_id % 2))]
            # Arrow length
            al = settings.TILESIZE * 0.8
            # Arrow point, back, side1, side2
            ap = (tc[0] + ad[0] * al / 2, tc[1] + ad[1] * al / 2)
            ab = (tc[0] - ad[0] * al / 2, tc[1] - ad[1] * al / 2)
            as1 = (ap[0] - ad[0] * al / 4 - ad[1] * al / 4, ap[1] - ad[1] * al / 4 + ad[0] * al / 4)
            as2 = (ap[0] - ad[0] * al / 4 + ad[1] * al / 4, ap[1] - ad[1] * al / 4 - ad[0] * al / 4)
            pygame.draw.line(self.image, settings.BLACK, ab, ap, 5)
            pygame.draw.line(self.image, settings.BLACK, ap, as1, 5)
            pygame.draw.line(self.image, settings.BLACK, ap, as2, 5)

    def update(self):
        """
            Updates the tile's position.
        """
        self.rect.x = settings.START[0] + (self.x * settings.TILESIZE)
        self.rect.y = settings.START[1] + (self.y * settings.TILESIZE)
        
    def click(self, mouse_pos):
        """
            Checks if the tile is clicked or not.
            
            Args:
                mouse_pos (tuple): Mouse position
            
            Returns:
                bool: True if the tile is clicked, False otherwise
        """
        return (self.rect.left <= mouse_pos[0] <= self.rect.right) \
            and (self.rect.top <= mouse_pos[1] <= self.rect.bottom)

    def right(self):
        """
            Checks if the tile's right side is empty or not.
            
            Returns:
                bool: True if the tile's right side is empty, False otherwise
        """
        return self.rect.x + settings.TILESIZE < settings.GAMESIZE * settings.TILESIZE + settings.START[0]

    def left(self):
        """
            Checks if the tile's left side is empty or not.
            
            Returns:
                bool: True if the tile's left side is empty, False otherwise
        """
        return self.rect.x - settings.TILESIZE >= 0 + settings.START[0]

    def up(self):
        """
            Checks if the tile's upside is empty or not.
            
            Returns:
                bool: True if the tile's upside is empty, False otherwise
        """
        return self.rect.y - settings.TILESIZE >= 0 + settings.START[1]

    def down(self):
        """
            Checks if the tile's downside is empty or not.
            
            Returns:
                bool: True if the tile's downside is empty, False otherwise
        """
        return self.rect.y + settings.TILESIZE < settings.GAMESIZE * settings.TILESIZE + settings.START[1]
