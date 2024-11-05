import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import numpy as np
import statistics
import os

def read_csv_to_list(file_path):
    numbers = []

    # Open the CSV file
    with open(file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Iterate through each row in the CSV file
        for row in csvreader:
            # Convert each value in the row to float (or int) and append to the numbers list
            for value in row:
                try:
                    # Convert to float, change to int if you only need integers
                    numbers.append(int(value))
                except ValueError:
                    print(f"Warning: Could not convert '{value}' to a number.")

    return numbers

def plot_frequency_bar_graph(numbers, title, min, maximum, outputFile):
    # Count the frequency of each individual number
    frequency = Counter(numbers)

    # Separate numbers (keys) and their corresponding frequencies (values)
    x_values = list(frequency.keys())
    y_values = list(frequency.values())

    # Create a bar graph for individual numbers
    plt.figure(figsize=(12, 6))
    plt.bar(x_values, y_values, color='blue', edgecolor='black', alpha=0.7)

    # Set limits for the x-axis to fit the range -512 to 512
    plt.xlim(min, maximum)

    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    plt.title(title)

    # Set y-axis ticks with a specific interval (0, 10, 20, ..., max frequency)
    max_y = max(y_values)  # Find the maximum y-value to set the upper limit
    plt.yticks(range(0, max_y + 10, 20))  # Adjust the step to your needs (here, 10)

    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)  # Add grid lines for better readability


    # Show the plot (optional, if you want to view it)
    plt.tight_layout()  # Adjust layout to make room for labels

    # Save the plot as PDF
    plt.savefig(outputFile)

def compute_average(solutions):
    return sum(solutions) / len(solutions)

def compute_standard_deviation(data : list):        
    data = np.array(data)
    return np.std(data)

def compute_range(solutions):
    return max(solutions) - min(solutions)

def compute_median(solutions):
    return statistics.median(solutions)

def get_solution_data(fileName):
    with open(fileName, 'r') as file:
        reader = csv.reader(file)

        times = []
        bestSolutions = []
        allSolutions = []

        # Skip the header row
        next(reader)

        row_count = 0  # To count the number of rows processed

        for row in reader:
            if not row:  # Skip empty rows
                continue  

            try:
                # Capture time and best solution
                times.append(int(row[0])) 
                bestSolutions.append(float(row[1]))  
                
                # Extend allSolutions with all 30 solutions in the row
                allSolutions.extend(float(x) for x in row[2:32])  
                
                row_count += 1
            except ValueError:
                continue  # Skip rows with ValueError

    return times, bestSolutions, allSolutions

def get_best_solution_data_by_function(solutions):
    schwefels, deJongOne, rosenBrock, rastrigin, griewangk = [], [], [], [], []
    sineEnvelopeSineWave, stretchVSineWave, ackleyOne, ackleyTwo, eggHolder = [], [], [], [], []

    functions = [schwefels, deJongOne, rosenBrock, rastrigin, griewangk,
                 sineEnvelopeSineWave, stretchVSineWave, ackleyOne, ackleyTwo, eggHolder]

    # Assuming each function has 30 solutions, we need to collect them accordingly
    for j in range(len(functions)):
        # Calculate the starting index for this function's solutions
        start_index = j * 30  # Start at 0 for the first function, 30 for the second, etc.
        functions[j].extend(solutions[start_index:start_index + 30])  # Append 30 solutions for each function

    return (schwefels, deJongOne, rosenBrock, rastrigin, 
            griewangk, sineEnvelopeSineWave, stretchVSineWave, 
            ackleyOne, ackleyTwo, eggHolder)

def get_improvement_data(fileName):
    
    with open(fileName, 'r') as file:
        reader = csv.reader(file)

        iterations = []
        newBestSolutions = []
        originalSolutions = []
        improvements = []

        # Skip the header row
        next(reader)

        for row in reader:
            # Check if the row is not empty
            if not row:  # If the row is empty
                continue  # Skip to the next iteration
            try:
                iterations.append(int(row[0]))  # Access row[0] safely
                newBestSolutions.append(float(row[1]))  # Access row[1] safely
                originalSolutions.append(float(row[2]))
                improvements.append(originalSolutions[-1] - newBestSolutions[-1])
            except ValueError:
                # Add placeholder values for both iterations and newBestSolutions
                iterations.append(None)  # or any other placeholder value
                newBestSolutions.append(None)  # or any other placeholder value
                originalSolutions.append(None)

    return iterations, newBestSolutions, originalSolutions, improvements
def write_improvement_data_to_file(outputName, iterations, improvedSolutions, originalSolutions, improvements, bestSolutions):
    
    with open(outputName, 'a') as file:
       
        file.write("\n---------------------------------LOCAL SEARCH ON BEST SOLUTIONS DATA---------------------------------\n")
        denom = len(bestSolutions)
        
        file.write(f"Number of best solutions we improved on: {len(improvedSolutions)} out of {denom}\n")
        file.write(f"Successfully improving rate: {len(improvedSolutions)/denom}\n")
        file.write(f"\nAverage improvement: {compute_average(improvements)}\n")
        file.write(f"Median improvement: {compute_median(improvements)}\n")
        file.write(f"Standard Deviation of improvement: {compute_standard_deviation(improvements)}\n")
        file.write(f"Range of improvement: {compute_range(improvements)}\n")
        file.write(f"Our best local search improvement: {max(improvements)}\n")
        
        file.write(f"Average number of iterations per improved solution: {compute_average(iterations)}\n")
        file.write(f"Median number of iterations per improvement: {compute_median(iterations)}\n")
        file.write(f"Standard Deviation of number of iterations per improvement: {compute_standard_deviation(iterations)}\n")
        file.write(f"Range of number of iterations per improvement: {compute_range(iterations)}\n")
        file.write(f"<Most iterations for an improvement: {max(iterations)}\n")

def write_solution_data_to_file(file_path, randTimes, uRandTimes, randAllSolutions, uRandAllSolutions, rBestSolutions, uBestSolutions):
    with open(file_path, 'w') as file:
        
        # Write time data for both matrices
        file.write("\n-----------------------------------TIME DATA---------------------------------------\n")

        file.write(f"Rand Average Solution time : {compute_average(randTimes)}\n")
        file.write(f"URand Average Solution time : {compute_average(uRandTimes)}\n")
        file.write(f"Rand Standard Deviation of times : {compute_standard_deviation(randTimes)}\n")
        file.write(f"URand Standard Deviation of times : {compute_standard_deviation(uRandTimes)}\n")
        file.write(f"Rand Range of times : {compute_range(randTimes)}\n")
        file.write(f"URand Range of times : {compute_range(uRandTimes)}\n")
        file.write(f"Rand Median of times : {compute_median(randTimes)}\n")
        file.write(f"URand Median of times : {compute_median(uRandTimes)}\n")

        # split up all of our solutions by their functions
        randSchwefelsAllSolutions, randDeJongsAllSolutions, randRosenBrockAllSolution, randRastrigin, randGriewangk, randSineEnvelopeSineWave, randStretchVSineWave, randAckleyOne, randAckleyTwo, randEggHolder = get_best_solution_data_by_function(randAllSolutions)
        uRandSchwefelsAllSolutions, uRandDeJongsAllSolutions, uRandRosenBrockAllSolution, uRandRastrigin, uRandGriewangk, uRandSineEnvelopeSineWave, uRandStretchVSineWave, uRandAckleyOne, uRandAckleyTwo, uRandEggHolder = get_best_solution_data_by_function(uRandAllSolutions)

        # Create a list of tuples for functions respective best solutions, and their lists of all solutions
        functions = [
            ("Schwefels Function", rBestSolutions[0], uBestSolutions[0], randSchwefelsAllSolutions, uRandSchwefelsAllSolutions),
            ("De Jong One Function", rBestSolutions[1], uBestSolutions[1], randDeJongsAllSolutions, uRandDeJongsAllSolutions),
            ("Rosen Brock Saddle Function", rBestSolutions[2], uBestSolutions[2], randRosenBrockAllSolution, uRandRosenBrockAllSolution),
            ("Rastrigin Function", rBestSolutions[3], uBestSolutions[3], randRastrigin, uRandRastrigin),
            ("Griewangk Function", rBestSolutions[4], uBestSolutions[4], randGriewangk, uRandGriewangk),
            ("Sine Envelope Sine Wave Function", rBestSolutions[5], uBestSolutions[5], randSineEnvelopeSineWave, uRandSineEnvelopeSineWave),
            ("Stretch VSine Wave Function", rBestSolutions[6], uBestSolutions[6], randStretchVSineWave, uRandStretchVSineWave),
            ("Ackley One Function", rBestSolutions[7], uBestSolutions[7], randAckleyOne, uRandAckleyOne),
            ("Ackley Two Function", rBestSolutions[8], uBestSolutions[8], randAckleyTwo, uRandAckleyTwo),
            ("Egg Holder Function", rBestSolutions[9], uBestSolutions[9], randEggHolder, uRandEggHolder)
        ]

        # Write all the different functions and their data
        for function_name, rBestSolution, uBestSolution, randSols, uRandSols in functions:
            file.write(f"\n---------------{function_name}-------------------\n")
            file.write(f"Rand Matrix Average Solution : {compute_average(randSols)}\n")
            file.write(f"URandom Matrix Average Solution : {compute_average(uRandSols)}\n\n")
            file.write(f"Rand Matrix Standard Deviation of Solutions : {compute_standard_deviation(randSols)}\n")
            file.write(f"URandom Matrix Standard Deviation of Solutions : {compute_standard_deviation(uRandSols)}\n\n")
            file.write(f"Rand Matrix Range of Solutions : {compute_range(randSols)}\n")
            file.write(f"URandom Matrix Range of Solutions : {compute_range(uRandSols)}\n\n")
            file.write(f"Rand Matrix Median of Solutions : {compute_median(randSols)}\n")
            file.write(f"URandom Matrix Median of Solutions : {compute_median(uRandSols)}\n\n")
            file.write(f"Rand Most Optimal Solution Found: {rBestSolution}\n")
            file.write(f"URandom Most Optimal Solution Found : {uBestSolution}\n")

# get our lists out of our CSVs.
randMatrix30List = read_csv_to_list("randMatrix30.csv")
uRandMatrix30List = read_csv_to_list("uRandomMatrix30.csv")
randMatrix32List = read_csv_to_list("randMatrix32.csv")
uRandMatrix32List = read_csv_to_list("uRandomMatrix32.csv")
randMatrix100List = read_csv_to_list("randMatrix100.csv")
uRandMatrix100List = read_csv_to_list("uRandomMatrix100.csv")
randMatrix500List = read_csv_to_list("randMatrix500.csv")
uRandMatrix500List = read_csv_to_list("uRandomMatrix500.csv")
randMatrix512List = read_csv_to_list("randMatrix512.csv")
uRandMatrix512List = read_csv_to_list("uRandomMatrix512.csv")

# plot our frequency bar graphs
output_folder = 'PRNG_Distributions'
os.makedirs(output_folder, exist_ok=True)
# make the graphs and save them in a new folder called PRNG_Distributions
plot_frequency_bar_graph(randMatrix30List, "[-30, 30] Random Matrix Frequency Bar Graph", -30, 30, os.path.join(output_folder, 'RandMatrix(-30_30).pdf'))
plot_frequency_bar_graph(uRandMatrix30List, "[-30, 30] URandom Matrix Frequency Bar Graph", -30, 30, os.path.join(output_folder, 'URandomMatrix(-30_30).pdf'))
plot_frequency_bar_graph(randMatrix32List, "[-32, 32] Random Matrix Frequency Bar Graph", -32, 32, os.path.join(output_folder, 'RandMatrix(-32_32).pdf'))
plot_frequency_bar_graph(uRandMatrix32List, "[-32, 32] URandom Matrix Frequency Bar Graph", -32, 32, os.path.join(output_folder, 'URandomMatrix(-32_32).pdf'))
plot_frequency_bar_graph(randMatrix100List, "[-100, 100] Random Matrix Frequency Bar Graph", -100, 100, os.path.join(output_folder, 'RandMatrix(-100_100).pdf'))
plot_frequency_bar_graph(uRandMatrix100List, "[-100, 100] URandom Matrix Frequency Bar Graph", -100, 100,os.path.join(output_folder, 'URandomMatrix(-100_100).pdf'))
plot_frequency_bar_graph(randMatrix500List, "[-500, 500] Random Matrix Frequency Bar Graph", -500, 500, os.path.join(output_folder, 'RandMatrix(-500_500).pdf'))
plot_frequency_bar_graph(uRandMatrix500List, "[-500, 500] URandom Matrix Frequency Bar Graph", -500, 500, os.path.join(output_folder, 'URandomMatrix(-500_500).pdf'))
plot_frequency_bar_graph(randMatrix512List, "[-512, 512] Random Matrix Frequency Bar Graph", -512, 512, os.path.join(output_folder, 'RandMatrix(-512_512).pdf'))
plot_frequency_bar_graph(uRandMatrix512List, "[-512, 512] URandom Matrix Frequency Bar Graph", -512, 512,os.path.join(output_folder, 'URandomMatrix(-512_512).pdf'))

# get our data out of our solution csv files
randTimes, randBestSolutions, randAllSolutions  = get_solution_data("randOutput.csv")
uRandTimes, uRandBestSolutions, uRandAllSolutions = get_solution_data("uRandomOutput.csv")

# Usage example
write_solution_data_to_file(os.path.join(output_folder, 'PRNGTesterOutputReport.txt'), randTimes, uRandTimes, randAllSolutions, uRandAllSolutions, randBestSolutions, uRandBestSolutions)

iterations, newBests, originalBests, improvements = get_improvement_data('improvements.csv')

write_improvement_data_to_file(os.path.join(output_folder, 'PRNGTesterOutputReport.txt'), iterations, newBests, originalBests, improvements, randBestSolutions)

print("Statistics Gathering Complete\nOutputs can be viewed in PRNG_Distributions folder")
print("Output statistics are in PRNGTesterOutputReport.txt")

os.system("rm *.csv")