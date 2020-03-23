import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess as sp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    #decl
    finished = False
    agentNum = 0
    results = []
    imgFile = "self-play-results.png"

    #init
    sp.call(["python", "init_agent.py", "--board-size", "9", ("agents/p_agent_99_" + str(agentNum) + ".h5")])
    if not os.path.exists("experiences"):
        os.makedirs("experiences")

    #loop
    while not finished:
        #info
        print("Loop " + str(agentNum + 1) + ":")

        #self-play
        sp.run(["python", "self_play.py", "--board-size", "9", "--learning-agent", ("agents/p_agent_99_" + str(agentNum) + ".h5"),\
            "-n", "5000", "--experience-out", ("experiences/p_exp_r" + str(agentNum + 1) + ".h5")])

        #train
        agentNum += 1
        sp.run(["python", "train_pg.py", "--learning-agent", ("agents/p_agent_99_" + str(agentNum - 1) + ".h5"),\
            "--agent-out", ("agents/p_agent_99_" + str(agentNum) + ".h5"), ("experiences/p_exp_r" + str(agentNum) + ".h5")])

        #eval
        bEvalOutAll = sp.run(["python", "eval_pg_bot.py", "--agent1", ("agents/p_agent_99_" + str(agentNum) + ".h5"),\
            "--agent2", "agents/p_agent_99_1.h5", "-n", "50"], stdout=sp.PIPE).stdout

        #parse
        evalOutAll = bEvalOutAll.decode()
        print(evalOutAll)
        eoa = evalOutAll.split("\n")
        evalOut = eoa[len(eoa) - 2] # the 2 here is because there is a \n at the end
        evalOut = evalOut[16:] # the 16 here are "Agent 1 record: "
        eo = evalOut.split("/")
        gamesWon = int(eo[0])
        results.append(gamesWon * 2)
        finished = gamesWon >= 45

        #graph
        fig = plt.figure()
        plt.plot(results, "k")
        if len(results) >= 5:
            epRange = list(range(1, len(results) + 1))
            fitVals = np.polyfit(epRange, results, 1)
            fit = np.poly1d(fitVals)
            plt.plot(fit(epRange), "r--")
        plt.title("% Games Won / # of Rounds")
        plt.ylabel("% Validation Games Won")
        plt.xlabel("Play-Train Cycles")
        fig.savefig(imgFile, dpi=200)
        plt.close(fig)

        #info
        print("Loop " + str(agentNum) + " score: " + str(gamesWon))

if __name__ == '__main__':
    main()