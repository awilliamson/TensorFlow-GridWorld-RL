from gridworld import *
from dqn import *

###
#
###
def serialise_gridworld( gw ):
    rtn = []
    for x in range(gw.height):
        for y in range(gw.width):
            v = gw.map[y][x].value

            # Anything other than a wall is walkable, default it down.
            if v is not 1:
                v = 0

            rtn.append(v)  # Flip columns and rows for pretty-printing

    for p in gw.player:
        rtn.append(p)

    return rtn

ACTIONS = [ Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT ]

if __name__ == "__main__":
    gw = GridWorld()
    dqn = DQN(input_shape=[None, len(serialise_gridworld(gw))], num_outputs=4)

    gw = GridWorld(5, 5)
    print(gw)

    #gw.move(Move.RIGHT)
    #print(gw.getReward())
    #print(gw)#

    #gw.move(Move.RIGHT)
    #print(gw.getReward())
    #print(gw)

#    gw.move(Move.DOWN)
#    print(gw.getReward())
#    print(gw)
#
#    gw.move(Move.DOWN)
#    print(gw.getReward())
#    print(gw)
#
#    gw.move(Move.DOWN)
#    print(gw.getReward())
#    print(gw)
#
#    gw.move(Move.DOWN)
 ##   print(gw.getReward())
#    print(gw)
#
#    gw.move(Move.RIGHT)
#    print(gw.getReward())
#    print(gw)
#
#    assert (gw.player == gw.goal)
#    print(gw.player, gw.goal)
#    print( serialise_gridworld( gw ) )

    history = []
    while gw.player != gw.goal :
        at = ACTIONS[ dqn.action([]) ]
        st = serialise_gridworld(gw)
        gw.move( at )
        st_1 = serialise_gridworld(gw)
        rt = gw.getReward()
        #print(str(at) + ",\t" + str(rt))
        #print( gw )

        # Transition St, At, Rt, St+1, Terminal
        history.append([serialise_gridworld(gw), at, rt, st_1, gw.player == gw.goal])
        dqn.store([serialise_gridworld(gw), at, rt, st_1, gw.player == gw.goal])

        dqn.train()

    #for x in history[:]:
    #    print(x)

    print( "Length of Move History: " + str(len(history)))
    print( "Length of Experience replay: " + str(len(dqn.experience_replay)))

    assert( history[-1] == dqn.experience_replay[-1] )
    assert( history[-50] == dqn.experience_replay[0] )

    #while( True ):
    #    dqn.train()
    #    print( "Update Count: ", dqn.target_q_network_update_count )