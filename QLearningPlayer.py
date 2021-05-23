import numpy as np
import ludopy

class QLearningPlayer():
    def __init__(self, filepath):
        self.Q = self.loadQTable(filepath)
        self.numOfPieces = 4
        self.numOfStates = 6 * self.numOfPieces
        self.numOfActions = 4

    def loadQTable(self,filepath):
        print("Loading Q Table.")
        return np.load(filepath)

    def getQValue(self, state, diceIdx, action):
        idx = state+[diceIdx]+[action]
        return self.Q[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5],idx[6],idx[7],idx[8],idx[9],idx[10],idx[11],idx[12],idx[13],idx[14],idx[15],idx[16],idx[17],idx[18],idx[19],idx[20],idx[21],idx[22],idx[23],idx[24],idx[25]]

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

    def getNextAction(self, state, dice, movePieces):
        diceIdx = dice - 1
        bestAction = movePieces[0]
        bestQValue = self.getQValue(state, diceIdx, bestAction)
        for action in movePieces:
            if self.getQValue(state,diceIdx, action) > bestQValue:
                bestAction = action
                bestQValue = self.getQValue(state,diceIdx,action)

        return bestAction


player = QLearningPlayer('BestQTable.npy') #Give the path to the QTable.
g = ludopy.Game()
gameNumber = 0
winners = []
winnersRate = []
numOfGames = 20000
while gameNumber < numOfGames:
    g.reset()
    gameNumber += 1
    there_is_a_winner = False
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

        if player_i == 0 and len(move_pieces) > 0:
            #This is the line that is important
            piece_to_move = player.getNextAction(player.getState(player_pieces,enemy_pieces), dice, move_pieces)
        else:
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
    winners.append(player_i)
    winnersRate.append([(winners.count(0) / gameNumber) * 100, (winners.count(1) / gameNumber) * 100, (winners.count(2) / gameNumber) * 100, (winners.count(3) / gameNumber) * 100])
    print("Players winrate:", winnersRate[-1],"Game nr:", gameNumber)

with open("FinalEval.txt", 'w+') as file:
    for rate in winnersRate:
        string = ''
        for r in rate:
            string += str(r) + " "
        string = string[0:-1]
        file.write(string + "\n")