from io import StringIO
import sys
import os

from tqdm import tqdm
from simulate_game import simulate_game

# First, import the ThreadPoolExecutor class from the concurrent.futures module
from concurrent.futures import ThreadPoolExecutor

# Get the number of CPU cores on the system
num_cores = os.cpu_count()
print(num_cores)
num_cores = 4

# Create the threadpool executor with the same number of workers as CPU cores
executor = ThreadPoolExecutor(max_workers=num_cores)

def main() -> None:
    results = []

    # Next, define the function that you want to run in parallel
    def run_file(player_1, player_2, time, board):
            res = os.popen('python simulate_game.py --first ' + player_1 + ' --second ' + player_2 + ' --time ' + time + ' --board boards\\' + board).read()
            return (player_1, player_2, time, board, res.splitlines()[-1])

    times = ["0.1", "0.5", "1", "5"]
    players = ["greedy_player", "team40_A1", "team40_A2"]
    boards = ["easy-2x2.txt","easy-3x3.txt","empty-3x3.txt","hard-3x3.txt","random-2x3.txt","random-3x3.txt","random-3x4.txt","random-4x4.txt"]
    repeats = 5

    win = 0
    loss = 0
    draw = 0

    # Submit the function to the executor to run in parallel
    for player in players:
        for time in times:
            for board in boards:
                for i in range(repeats):
                    result = executor.submit(run_file, "team40_A3", player, time, board)
                    results.append(result)
                    result = executor.submit(run_file, player, "team40_A3", time, board)
                    results.append(result)
    
    # Get the number of jobs that are still running or pending
    jobs_remaining = len([r for r in results if not r.done()])

    # Create a progress bar using tqdm
    pbar = tqdm(total=len(results), desc="Processing Results", leave=True)

    # Open the file for writing
    with open('results_A3.txt', 'w') as outfile:
        # Get the results from the function calls
        for result in results:
            tqdm.write(str(result.result()))
            if result.result()[0] == "team40_A3":
                if result.result()[4].find("Player 1") != -1:
                    win += 1
                elif result.result()[4].find("draw") != -1:
                    draw += 1
                else:
                    loss += 1
            else:
                if result.result()[4].find("Player 2") != -1:
                    win += 1
                elif result.result()[4].find("draw") != -1:
                    draw += 1
                else:
                    loss += 1
            tqdm.write(str(win) + "/" + str(draw) + "/" + str(loss))
            # Write the result to the file
            outfile.write(str(result.result()) + '\n')
                    # Decrement the number of jobs remaining
            jobs_remaining -= 1
            # Update the progress bar
            pbar.update(1)
        outfile.write(str(win) + "/" + str(draw) + "/" + str(loss))

if __name__ == '__main__':
    main()