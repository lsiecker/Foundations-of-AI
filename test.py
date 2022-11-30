from io import StringIO
import sys
import os
from simulate_game import simulate_game

def main():

    def run_file(player_1, player_2, time, board):
        res = os.popen('python simulate_game.py --first ' + player_1 + ' --second ' + player_2 + ' --time ' + time + ' --board boards\\' + board).read()
        print(player_1, player_2, time, board, res.splitlines()[-1])


        
    
    times = ["0.1", "0.5", "1", "5"]
    players = ["greedy_player", "random_player"]
    boards = ["easy-2x2.txt","easy-3x3.txt","empty-3x3.txt","hard-3x3.txt","random-2x3.txt","random-3x3.txt","random-3x4.txt","random-4x4.txt"]

    for player in players:
        for time in times:
            for board in boards:
                run_file("team40_A1", player, time, board)
                run_file(player, "team40_A1", time, board)
    
    

if __name__ == '__main__':
    main()