class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        print(self.width, self.height)

        # Create 2D Array, filled with 0's.
        self.map = [[0 for _ in range(self.width)] for _ in range(self.height)]


if __name__ == "__main__":
    gw = GridWorld( 5, 5 )