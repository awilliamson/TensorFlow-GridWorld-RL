from gridworld import *
from dqn import *

if __name__ == "__main__":
    gw = GridWorld()
    dqn = DQN()

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

    print(gw.player, gw.goal)