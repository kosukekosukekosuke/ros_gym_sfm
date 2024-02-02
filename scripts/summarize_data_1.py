import glob
import csv
import numpy as np
import os

input_path = "/home/amsl/catkin_ws/src/ros_gym_sfm/data/experiment_1/"
output_path = "/home/amsl/catkin_ws/src/ros_gym_sfm/data/"
output_file_name = "thesis_data_1.csv"

files = glob.glob(input_path + "*")
for file in files:
    print(file)
    input_file_name = file

    # open the csv file
    with open(input_file_name) as f:
        reader = csv.reader(f)
        csv_data = [row for row in reader]

    # convert str to float and create an array for each column
    converted_data_1 = []  # success judgment
    converted_data_2 = []  # goal time
    for i in range(len(csv_data)):
        for j in range(len(csv_data[i])):
            float_data = float(csv_data[i][j])
            if j == 0:
                converted_data_1.append(float_data)
            if j == 1:
                if float_data != 0:
                    converted_data_2.append(float_data)

    # calculate success rate
    sum_success = 0
    success_rate = 0
    for i in range(len(converted_data_1)):
        sum_success = sum_success + converted_data_1[i]
    success_rate = sum_success / len(converted_data_1)

    # calculate average goal time
    sum_goal_time = 0
    average_goal_time = 0
    for i in range(int(sum_success)):
        sum_goal_time = sum_goal_time + converted_data_2[i]
    average_goal_time = sum_goal_time / sum_success

    # calculate standard deviation (and variance) of the goal times
    variance_goal_time = np.var(converted_data_2)
    standard_deviation = np.std(converted_data_2)

    print(f"{success_rate},{average_goal_time},{standard_deviation}")

    # write data to a file
    header = ['file path', 'Success Rate', 'Average Goal Time', 'Standard Deviation']
    body = [input_file_name, success_rate, average_goal_time, standard_deviation]
        # check if there is a csv file
    is_file = os.path.isfile(output_path + output_file_name)
    if is_file:
        print(f"'{output_file_name}' exists.")
        with open(output_path + output_file_name, 'a') as f:
            writer = csv.writer(f)
            
            writer.writerow(body)
    else:
        print(f"'{output_file_name}' does not exist.")
        with open(output_path + output_file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(body)


