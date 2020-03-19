import os
import subprocess as sp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    #decl
    finished = False
    agentNum = 0

    #init
    sp.call(["python", "init_agent.py", "--board-size", "9", ("agents/p_agent_99_" + str(agentNum) + ".h5")])
    if not os.path.exists("experiences"):
        os.makedirs("experiences")

    #loop
    while not finished:
        #info
        print("Loop " + str(agentNum + 1) + ":")

        #self-play
        sp.call(["python", "self_play.py", "--board-size", "9", "--learning-agent", ("agents/p_agent_99_" + str(agentNum) + ".h5"),\
            "-n", "1000", "--experience-out", ("experiences/p_exp_r" + str(agentNum + 1) + ".h5")])

        #train
        agentNum += 1
        sp.call(["python", "train_pg.py", "--learning-agent", ("agents/p_agent_99_" + str(agentNum - 1) + ".h5"),\
            "--agent-out", ("agents/p_agent_99_" + str(agentNum) + ".h5"), ("experiences/p_exp_r" + str(agentNum) + ".h5")])

        #eval
        bEvalOutAll = sp.check_output(["python", "eval_pg_bot.py", "--agent1",\
            ("agents/p_agent_99_" + str(agentNum) + ".h5"), "--agent2", "agents/p_agent_99_1.h5", "-n", "25"])

        #parse
        evalOutAll = bEvalOutAll.decode()
        print(evalOutAll)
        eoa = evalOutAll.split("\n")
        evalOut = eoa[len(eoa) - 2] # the 2 here is because there is a \n at the end
        evalOut = evalOut[16:] # the 16 here are "Agent 1 record: "
        eo = evalOut.split("/")
        gamesWon = int(eo[0])
        finished = gamesWon >= 20

        #info
        print("Loop " + str(agentNum) + " score: " + str(gamesWon))

if __name__ == '__main__':
    main()