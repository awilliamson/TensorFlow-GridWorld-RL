from gridworld import *
from dqn import *

def serialise_gridworld( gw ):
    rtn = []
    for x in range(gw.height):
        for y in range(gw.width):
            if gw.player == (y, x):
                rtn.append('P')  # Visualise where the player is
            else:
                rtn.append(gw.map[y][x])  # Flip columns and rows for pretty-printing
    return rtn

if __name__ == "__main__":
    gw = GridWorld()
    dqn = DQN(input_shape=[None, len(serialise_gridworld(gw)) ], num_outputs=4)

    gw = GridWorld(5, 5)
    print(gw)

    gw.move(Move.RIGHT)
    print(gw.getReward())
    print(gw)

    gw.move(Move.RIGHT)
    print(gw.getReward())
    print(gw)

    gw.move(Move.DOWN)
    print(gw.getReward())
    print(gw)

    gw.move(Move.DOWN)
    print(gw.getReward())
    print(gw)

    gw.move(Move.DOWN)
    print(gw.getReward())
    print(gw)

    gw.move(Move.DOWN)
    print(gw.getReward())
    print(gw)

    gw.move(Move.RIGHT)
    print(gw.getReward())
    print(gw)

    assert (gw.player == gw.goal)
    print(gw.player, gw.goal)