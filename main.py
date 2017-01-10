from gridworld import *
from dqn import *

available_moves = [ Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT ]

def serialise_gridworld( gw, pretty=False ):
    rtn = []
    for x in range(gw.height):
        for y in range(gw.width):
                if gw.player == (y, x):
                    rtn.append( 'P' if pretty else 4 )  # Visualise where the player is
                else:
                    rtn.append( gw.map[y][x] if pretty else gw.map[y][x].value)  # Flip columns and rows for pretty-printing
    return np.reshape( rtn, [1,len( rtn )] )

def init():
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

if __name__ == "__main__":
    gw = GridWorld(5, 5)
    dqn = DQN(input_shape=gw.width*gw.height, num_outputs=4)

    # For debugging GridWorld. This ensures we have a functioning world, and optimal policy!
    init()

    print( serialise_gridworld(gw, False))

    s = serialise_gridworld(gw)
    print( "Len", len(s) )
    a = dqn.get_action(s)[0]
    print("[ACTION]:", available_moves[a] )

    r = gw.getReward()

    # Copy the gridworld, apply action, observe output.
    gwt2 = gw
    gwt2.move( available_moves[a] )
    st = serialise_gridworld(gwt2)

    dqn.add_experience(s, a, r, st)
    dqn.training()

