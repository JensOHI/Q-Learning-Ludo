import ludopy
import numpy as np
import itertools
import time
import textwrap
from copy import deepcopy

import matplotlib.pyplot as plt

#https://ieeexplore-ieee-org.proxy1-bib.sdu.dk/document/6031999
#Complexity analysis and playing strategies for Ludo and its variant race games

#https://stackoverflow.com/questions/36459969/python-convert-list-to-dictionary-with-indexes-as-values
#https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
#https://stackoverflow.com/questions/3099987/generating-permutations-with-repetitions
#https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements

#https://stackoverflow.com/questions/48568717/how-to-encode-with-actual-bits-in-python?fbclid=IwAR3VWDgBgqbpdVPwOI4AuvZSxOUJTwGO1XaeSqsp5GXgkupfKTE6UoUlzxw

EPSILON = 0.15


class QLerning():
    def __init__(self, discountFactor_ = 0.95, lerningRate_ = 0.1, QThreshold_ = 0.001):
        #QLerning parameters
        self.discountFactor = discountFactor_     #gamma
        self.lerningRate = lerningRate_           #alpha
        self.epsilon = EPSILON
        self.QThreshold = QThreshold_                   #Stopping criteria when training should stop

        #Initializing Q table
        self.numOfPieces = 4
        self.numOfStates = 6 * self.numOfPieces
        self.numOfActions = 4
        p1_home = p1_safe = p1_vulnerable = p1_attacking = p1_finishLine = p1_finished = \
            p2_home = p2_safe = p2_vulnerable = p2_attacking = p2_finishLine = p2_finished = \
            p3_home = p3_safe = p3_vulnerable = p3_attacking = p3_finishLine = p3_finished = \
            p4_home = p4_safe = p4_vulnerable = p4_attacking = p4_finishLine = p4_finished = 2
        self.numOfDiceSides = 6
        self.Q = np.random.rand(p1_home, p1_safe, p1_vulnerable, p1_attacking, p1_finishLine, p1_finished,p2_home, p2_safe, p2_vulnerable, p2_attacking, p2_finishLine, p2_finished, p3_home, p3_safe, p3_vulnerable, p3_attacking, p3_finishLine, p3_finished,p4_home, p4_safe, p4_vulnerable, p4_attacking, p4_finishLine, p4_finished, self.numOfDiceSides, self.numOfActions)
        print(self.Q.shape)
        self.Q *= 0.001
        self.setPrevQ()

    def setPrevQ(self):
        self.prevQ = deepcopy(self.Q)

    def saveGameAsVideo(self,game):
        game.save_hist_video(f"BestGame.mp4")

    def saveQTable(self):
        print("Saving Q Table.")
        with open("BestQTable.npy", 'wb') as file:
            np.save(file,self.Q)

    def loadQTable(self):
        print("Loading Q Table.")
        self.Q = np.load("BestQTable.npy")

    def getLargestQValueDifference(self):
        return np.amax(self.Q-self.prevQ)
        #return np.amax(np.maximum(self.Q,self.prevQ))

    def getQValue(self, state, diceIdx, action):
        idx = state+[diceIdx]+[action]
        return self.Q[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5],idx[6],idx[7],idx[8],idx[9],idx[10],idx[11],idx[12],idx[13],idx[14],idx[15],idx[16],idx[17],idx[18],idx[19],idx[20],idx[21],idx[22],idx[23],idx[24],idx[25]]

    def setQValue(self,state,diceIdx,action,value):
        idx = state + [diceIdx] + [action]
        self.Q[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5],idx[6],idx[7],idx[8],idx[9],idx[10],idx[11],idx[12],idx[13],idx[14],idx[15],idx[16],idx[17],idx[18],idx[19],idx[20],idx[21],idx[22],idx[23],idx[24],idx[25]] += value

    def getState(self, playerPieces, enemyPieces):
        def distanceBetweenTwoPieces(piece, enemy, i):
            if enemy == 0 or enemy >= 53 or piece == 0 or piece >= 53:
                return 1000
            enemy_relative_to_piece = (enemy + 13 * i) % 52
            if enemy_relative_to_piece == 0: enemy_relative_to_piece = 52
            distances = [enemy_relative_to_piece - piece, (enemy_relative_to_piece - 52) - piece]
            return distances[np.argmin(list(map(abs,distances)))]

        HOME = 0
        SAFE = 1
        VULNERABLE = 2
        ATTACKING = 3
        FINISHLINE = 4
        FINISHED = 5

        home = [0]
        globes = [1, 9, 14, 22, 27, 35, 40, 48]
        unsafe_globes = [14, 27, 40]

        state = []
        for playerPiece in playerPieces:
            pieceState = [0] * (int)(self.numOfStates / self.numOfPieces)

            #Calculating the relative distance of all the enemy pieces to the players piece
            distanceToEnemy = []
            for i, enemy in enumerate(enemyPieces):
                for enemyPiece in enemy:
                    distanceToEnemy.append(distanceBetweenTwoPieces(playerPiece, enemyPiece, i + 1))

            if playerPiece in home:
                pieceState[HOME] = 1

            if playerPiece in globes:
                pieceState[SAFE] = 1

            vulnerable = any([-6 <= relativePosition < 0 for relativePosition in distanceToEnemy])
            if (vulnerable and playerPiece not in globes) or playerPiece in unsafe_globes: pieceState[VULNERABLE] = 1

            attacking = any([0 < relativePosition <= 6 for relativePosition in distanceToEnemy])
            if attacking: pieceState[ATTACKING] = 1

            if playerPiece >= 53:
                pieceState[FINISHLINE] = 1

            if playerPiece == 59:
                pieceState[FINISHED] = 1

            state += pieceState
        return state

    def getNextAction(self, state, dice):
        diceIdx = dice - 1
        bestAction = self.movePieces[0]
        bestQValue = self.getQValue(state, diceIdx, bestAction)
        for action in self.movePieces:
            if self.getQValue(state,diceIdx, action) > bestQValue:
                bestAction = action
                bestQValue = self.getQValue(state,diceIdx,action)

        if np.random.uniform(0, 1) < (1 - self.epsilon): #This is made so it takes a random choice no matter what epsilon % of the time.
            return bestAction
        else:
            return np.random.choice(self.movePieces)

    def getNextState(self, playerPieces_, enemyPieces_, action, dice): #Maybe it be easier just tell the game what move and get information for that.
        playerPieces = deepcopy(playerPieces_)
        enemyPieces = deepcopy(enemyPieces_)
        stars = [5, 12, 18, 25, 31, 38, 44, 51]
        starsRemoveExtra = [0, 1, 0, 1, 0, 1, 0, 0]
        globes = [1, 9, 14, 22, 27, 35, 40, 48] # 53 is not inserted here, because if player piece lands on it and an enemy is there. The enemy will be send home.

        if playerPieces[action] == 0:
            playerPieces[action] = 1

        elif playerPieces[action] + dice > 59:
            goalDiff = 59 - playerPieces[action]
            playerPieces[action] = playerPieces[action] - dice + goalDiff

        elif (playerPieces[action] + dice) in stars:
            if playerPieces[action] + dice == 51:
                playerPieces[action] = 59
            else:
                idx = stars.index(playerPieces[action] + dice)
                playerPieces[action] += dice + 7 - starsRemoveExtra[idx]

        else:
            playerPieces[action] += dice

        for i,enemy in enumerate(enemyPieces):
            for p, enemyPiece in enumerate(enemy):
                enemyPiecesRelativeToPiece = enemyPiece + 13 * (i + 1) % 52
                if enemyPiecesRelativeToPiece == 0: enemyPiecesRelativeToPiece = 52
                if playerPieces[action] == enemyPiecesRelativeToPiece and playerPieces[action] <= 53 and playerPieces[action] > 0 and enemyPiece > 0 and enemyPiece <= 53:
                    if playerPieces[action] in globes:
                        playerPieces[action] = 0
                    elif enemyPieces[i].tolist().count(enemyPiece) > 1:
                        playerPieces[action] = 0
                    else:
                        enemyPieces[i][p] = 0
        return self.getState(playerPieces, enemyPieces)

    def avgNextStateQValues(self, nextState):
        qValues = []
        for dice in range(self.numOfDiceSides):
            for action in range(self.numOfActions):
                qValues.append(self.getQValue(nextState,dice,action))
        return np.average(qValues)

    def getReward(self, state, action, nextState):
        def changeBetweenStates(state_, nextState_):
            change = []
            for i in range(len(state_)):
                if state_[i] < nextState_[i]:
                    change.append(1)
                elif state_[i] > nextState_[i]:
                    change.append(-1)
                else:
                    change.append(0)
            return change

        flip = -1
        rewardHome = 0.15 * flip  # Bad going from not home -> home
        rewardSafe = 0.10
        rewardVulnerable = 0.10 * flip  # Bad going from not being vulnerable -> being vulnerable
        rewardAttacking = 0.25
        rewardFinishLine = 0.05
        rewardFinish = 0.35
        #even = 1/6
        #rewardHome = even * flip  # Bad going from not home -> home
        #rewardSafe = even
        #rewardVulnerable = even * flip  # Bad going from not being vulnerable -> being vulnerable
        #rewardAttacking = even
        #rewardFinishLine = even
        #rewardFinish = even
        #rewardHome = 0.4482421875 * flip  # Bad going from not home -> home
        #rewardSafe = 0.2578125
        #rewardVulnerable = 0.4052734375 * flip  # Bad going from not being vulnerable -> being vulnerable
        #rewardAttacking = 0.1923828125
        #rewardFinishLine = 0.0576171875
        #rewardFinish = 0.494140625
        rewardVector = np.asarray([rewardHome,rewardSafe,rewardVulnerable,rewardAttacking,rewardFinishLine,rewardFinish]*self.numOfActions)
        return np.dot(np.asarray(changeBetweenStates(state,nextState)), rewardVector)


    def saveInformation(self,winners, cumulativeReward_, nrOfTurns_, evaluateWinners_, evaluateCumulativeReward_, evaluateNrOfTurns_):
        winnersCumulativeWinrate = []
        for i in range(1,len(winners)):
            string = str((winners[0:i].count(0) / i)*100) + " " + str((winners[0:i].count(1) / i)*100) + " " + str((winners[0:i].count(2) / i)*100) + " " + str((winners[0:i].count(3) / i)*100)
            winnersCumulativeWinrate.append(string)
        with open("information/trainingWinners.txt", 'w+') as file:
            for line in winnersCumulativeWinrate:
                file.write(line+"\n")

        with open("information/trainingReward.txt", 'w+') as file:
            for r in cumulativeReward_:
                file.write(str(r) + "\n")

        with open("information/trainingTurns.txt", 'w+') as file:
            for t in nrOfTurns_:
                file.write(str(t) + "\n")

        with open("information/evaluateWinners.txt", 'w+') as file:
            for w in evaluateWinners_:
                w = list(w)
                string = str((w.count(0) / len(w)) * 100) + " " + str((w.count(1) / len(w)) * 100) + " " + str((w.count(2) / len(w)) * 100) + " " + str((w.count(3) / len(w)) * 100)
                file.write(string+"\n")

        with open("information/evaluateReward.txt", 'w+') as file:
            for r in evaluateCumulativeReward_:
                string = ''
                for element in list(r):
                    string += str(element) + " "
                string = string[0:-1]
                file.write(string + "\n")

        with open("information/evaluateTurns.txt", 'w+') as file:
            for t in evaluateNrOfTurns_:
                string = ''
                for element in list(t):
                    string += str(element) + " "
                string = string[0:-1]
                file.write(string + "\n")

    def train(self):
        game = ludopy.Game()

        episodes = 0
        evaluateEpisode = 0
        GamesToPlay_ = 30001
        GamesToPlay = GamesToPlay_
        gamesToEvaluate = 120
        evaluateAfter = 200

        evaluateWinners = []
        evaluateCumulativeReward = []
        evaluateNrOfTurns = []

        winners = []
        nrOfTurns_ = []
        cumulativeReward_ = []
        startTime = time.time()
        while GamesToPlay > 0:
            game.reset()
            winnerFound = False
            turnNumber = 1
            cumulativeReward = 0
            while not winnerFound:
                (dice, self.movePieces, playerPieces, enemyPieces, playerIsAWinner, winnerFound), playerIdx = game.get_observation()

                if len(self.movePieces): #No need to do QLerning with we can't make a move.
                    # ----------------- INSERT Q-LERNING HERE --------------------------
                    state = self.getState(playerPieces, enemyPieces)
                    action = self.getNextAction(state, dice)
                    nextState = self.getNextState(playerPieces, enemyPieces, action, dice)
                    #print(self.getPiecesNumbersString(state), self.getPiecesNumbersString(nextState), dice)

                    avgNextStateQValues = self.avgNextStateQValues(nextState)
                    reward = self.getReward(state, action, nextState)
                    cumulativeReward += reward
                    qValueChange = self.lerningRate * (reward + self.discountFactor * avgNextStateQValues - self.getQValue(state,dice-1,action))
                    self.setQValue(state,dice-1,action,qValueChange)
                    pieceToMove = action
                    # --------------------------------------------------------------------
                else:
                    pieceToMove = -1
                if(playerIdx == 0):
                    pass
                else: #If it is the one of the other 3 players turn, then just make a random move.
                    if len(self.movePieces):
                        pieceToMove = self.movePieces[np.random.randint(0, len(self.movePieces))]
                    else:
                        pieceToMove = -1

                _, _, _, _, _, winnerFound = game.answer_observation(pieceToMove)

                turnNumber += 1

            if episodes % evaluateAfter == 0 and evaluateEpisode < gamesToEvaluate:
                evaluateEpisode += 1
                evaluateWinners.append(playerIdx)
                evaluateCumulativeReward.append(cumulativeReward)
                evaluateNrOfTurns.append(turnNumber)
                print("Evaluate Episode:", evaluateEpisode)
                self.epsilon = 0
            else:
                self.epsilon = EPSILON
                evaluateEpisode = 0

                GamesToPlay -= 1

                winners.append(playerIdx)
                episodes += 1
                nrOfTurns_.append(turnNumber)
                cumulativeReward_.append(cumulativeReward)
                print("Episode:",episodes,"Turns:",turnNumber,"Cumulative reward:",cumulativeReward, "Winrate:",(winners.count(0) / episodes)*100, "%")

        self.saveQTable()
        print("Largest Q diff:",self.getLargestQValueDifference())
        print("Took:",time.time()-startTime,"seconds.")
        print("Took on avg:",np.average(nrOfTurns_),"turns.","Total turns:",np.sum(nrOfTurns_))
        print("Avg cumulative reward:",np.average(cumulativeReward_))
        print("Player 1:", (winners.count(0) / episodes)*100, "%")
        print("Player 2:", (winners.count(1) / episodes)*100, "%")
        print("Player 3:", (winners.count(2) / episodes)*100, "%")
        print("Player 4:", (winners.count(3) / episodes)*100, "%")

        evaluateWinners_= np.asarray(evaluateWinners).reshape(-1,gamesToEvaluate)
        numberOfWins = np.count_nonzero(evaluateWinners_ == 0, axis=1)
        winrate = (numberOfWins / gamesToEvaluate) * 100.0
        print("Player 1 (evaluate):", winrate)

        evaluateCumulativeReward_ = np.asarray(evaluateCumulativeReward).reshape(-1,gamesToEvaluate)
        evaluateNrOfTurns_ = np.asarray(evaluateNrOfTurns).reshape(-1, gamesToEvaluate)

        self.saveInformation(winners, cumulativeReward_, nrOfTurns_, evaluateWinners_, evaluateCumulativeReward_, evaluateNrOfTurns_)

        return game






qlerning = QLerning()
bestGame = qlerning.train()
qlerning.saveGameAsVideo(bestGame)








