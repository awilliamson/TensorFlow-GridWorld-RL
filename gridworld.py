from enum import Enum # Requires package enum34 as Python2.7 does not include this!

# Enums for Tile representation
class Tile( Enum ):
    free = 0
    wall = 1
    player = 2
    start = 3
    goal = 4

    def __str__(self):
        return str(self.name.capitalize()[0])

    def __repr__(self):
        return str(self.name.capitalize()[0])

class GridWorld:

    def __init__(self, width=5, height=5, start=(0,0), goal=(3,4), walls=(
        (3,0),
        (1,1),
        (0,2),(1,2),(3,2),
        (3,3),(4,3),
        (1,4),(4,4)
    )

    ):
        self.width = width
        self.height = height

        print(self.width, self.height)

        # Create 2D Array, filled with 0's.
        self.map = [[Tile.free for _ in range(self.height)] for _ in range(self.width)]

        self.start = start
        self.goal = goal

        self.walls = walls
        for w in walls:
            self.map[w[0]][w[1]] = Tile.wall

        self.player = self.start  # Player position in x, y


        self.map[self.start[0]][self.start[1]] = Tile.start
        self.map[self.player[0]][self.player[1]] = Tile.player
        self.map[self.goal[0]][self.goal[1]] = Tile.goal

    def __str__(self):
        ret = ""
        for x in range( self.height ):
            ret += "["
            for y in range( self.width ):
                ret += str(self.map[ y ][ x ]) #Flip columns and rows for pretty-printing
            ret += "]\n"
        return ret

    def move(self, action):
        if not isinstance(action, tuple):
            raise TypeError("Action must be tuple representing ( x, y ) movement.")

        _x, _y = tuple( [sum(x) for x in zip(self.player, action)] )
        valid = 0 <= _x < self.height and 0 <= _y < self.width and self.map[ _x ][ _y ] is not Tile.wall

        if valid is True:
            self.map[_x][_y] = Tile.player
            self.map[ self.player[0]][self.player[1]] = Tile.free

            self.player = (_x,_y)

        return valid

if __name__ == "__main__":
    gw = GridWorld( 5, 5 )
    print(gw)

    print(gw.move((3,0)))

    print(gw)