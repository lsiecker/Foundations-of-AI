from io import StringIO
import sys
import os
from simulate_game import simulate_game

def main():

    def run_file(player_1, player_2, time, board):
        res = os.popen('python simulate_game.py --first ' + player_1 + ' --second ' + player_2 + ' --time ' + time + ' --board boards\\' + board).read()
        print(player_1, player_2, time, board, res.splitlines()[-1])

    times = ["0.1", "0.5", "1", "5"]
    times = ["0.1", "0.5"]
    # players = ["random_player", "greedy_player"]
    players = ["random_player"]
    # boards = ["easy-2x2.txt","easy-3x3.txt","empty-2x2.txt","empty-2x3.txt","empty-3x3.txt","empty-3x4.txt","empty-4x4.txt",
    #     "hard-3x3.txt","random-2x3.txt","random-3x3.txt","random-3x4.txt","random-4x4.txt"]
    boards = ["empty-3x3.txt","empty-3x4.txt","hard-3x3.txt","random-3x3.txt","random-3x4.txt"]

    for player in players:
        for time in times:
            for board in boards:
                run_file("team40_A1", player, time, board)
                run_file(player, "team40_A1", time, board)
    
    

if __name__ == '__main__':
    main()

# 3x3 emp(ty board 0.1 seconds
# from contextlib import redirect_stdout
# import os

# with open('test.txt', 'w') as f:
#     with redirect_stdout(f):
#         print(os.system("python simulate_game.py --first team40_A1 --second greedy_player --time 0.5 --board boards\empty-2x2.txt"))
# python simulate_game.py --first greedy_player --second team40_A1 --time 0.1 --board boards\empty-3x3.txt

# # 3x3 empty board 0.5 seconds
# python simulate_game.py --first team40_A1 --second greedy_player --time 0.5 --board boards\empty-3x3.txt
# python simulate_game.py --first greedy_player --second team40_A1 --time 0.5 --board boards\empty-3x3.txt

# # 3x3 empty board 1 seconds
# python simulate_game.py --first team40_A1 --second greedy_player --time 1 --board boards\empty-3x3.txt
# python simulate_game.py --first greedy_player --second team40_A1 --time 1 --board boards\empty-3x3.txt

# # 3x3 empty board 5 seconds
# python simulate_game.py --first team40_A1 --second greedy_player --time 5 --board boards\empty-3x3.txt
# python simulate_game.py --first greedy_player --second team40_A1 --time 5 --board boards\empty-3x3.txt