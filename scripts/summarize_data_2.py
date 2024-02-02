import glob
import csv
import matplotlib.pyplot as plt

input_path = "/home/amsl/catkin_ws/src/ros_gym_sfm/data/experiment_2"
input_directory = "/actor_"
# input_directory_method = ["new_"]
input_directory_method = ["new_", "old_"]
input_directory_num = ["1"]
# input_directory_num = ["1", "5", "10", "15"]

d_size = 10

predicted_len = 3

plot_size = 20

def calc_difference(pose_list, actor_num):
    d_list = []
    for i in range(int(len(pose_list)/2)):
        d = (pose_list[i+actor_num] - pose_list[i])*d_size
        d_list.append(d)
    return d_list

def predicted_pose(pose_list, d_list):
    predicted_pose_list = []
    for i in range(len(d_list)):
        for j in range(predicted_len):    
            predicted_pose = pose_list[i] + d_list[i]*(j+1)
            predicted_pose_list.append(predicted_pose)
    return predicted_pose_list

for d_method in input_directory_method:
    for d_num in input_directory_num:
        files = glob.glob(input_path + input_directory + d_method + d_num + "/*")
        count = 0
        for f_num, input_file_name in  enumerate(files):
            print("flie_name:", input_file_name)
            # open the csv file
            with open(input_file_name) as f:
                reader = csv.reader(f)
                csv_data = [row for row in reader]

            # convert str to float and create an array for each column, plot a scatter plot
                # agent pose
            plt.rcParams["font.size"] = 30

            if "old_1_2_agent_pose" in input_file_name:
                print(-111)
                old_agent_x_list = []
                old_agent_y_list = []
                for i in range(len(csv_data)):
                    for j in range(len(csv_data[i])):
                        float_data = float(csv_data[i][j])
                        if j == 0:
                            old_agent_x_list.append(float_data)
                        if j == 1:
                            old_agent_y_list.append(float_data)
                # agent trajectory
            if "old_1_2_agent_trajectory" in input_file_name:
                print(-222)
                old_agent_trajectory_x_list = []
                old_agent_trajectory_y_list = []
                for i in range(len(csv_data)):
                    if i%20 == 0:
                        for j in range(len(csv_data[i])):
                            float_data = float(csv_data[i][j])
                            if j == 0:
                                old_agent_trajectory_x_list.append(float_data)
                            if j == 1:
                                old_agent_trajectory_y_list.append(float_data)
                # actor pose
            if "old_1_2_actor_pose" in input_file_name:
                print(-333)
                old_actor_x_list = []
                old_actor_y_list = []
                for i in range(len(csv_data)):
                    for j in range(len(csv_data[i])):
                        float_data = float(csv_data[i][j])
                        if j == 0:
                            old_actor_x_list.append(float_data)
                        if j == 1:
                            old_actor_y_list.append(float_data)
            # actor trajectory
            if "old_1_2_actor_trajectory" in input_file_name:
                print(-444)
                old_actor_trajectory_x_list = []
                old_actor_trajectory_y_list = []
                for i in range(len(csv_data)):
                    if i%20 == 0:
                        for j in range(len(csv_data[i])):
                            float_data = float(csv_data[i][j])
                            if j == 0:
                                old_actor_trajectory_x_list.append(float_data)
                            if j == 1:
                                old_actor_trajectory_y_list.append(float_data) 

            if "new_1_2_agent_pose" in input_file_name:
                print(111)
                new_agent_x_list = []
                new_agent_y_list = []
                for i in range(len(csv_data)):
                    for j in range(len(csv_data[i])):
                        float_data = float(csv_data[i][j])
                        if j == 0:
                            new_agent_x_list.append(float_data)
                        if j == 1:
                            new_agent_y_list.append(float_data)
                # agent trajectory
            if "new_1_2_agent_trajectory" in input_file_name:
                print(222)
                new_agent_trajectory_x_list = []
                new_agent_trajectory_y_list = []
                for i in range(len(csv_data)):
                    if i%20 == 0:
                        for j in range(len(csv_data[i])):
                            float_data = float(csv_data[i][j])
                            if j == 0:
                                new_agent_trajectory_x_list.append(float_data)
                            if j == 1:
                                new_agent_trajectory_y_list.append(float_data)
                # actor pose
            if "new_1_2_actor_pose" in input_file_name:
                print(333)
                new_actor_x_list = []
                new_actor_y_list = []
                for i in range(len(csv_data)):
                    for j in range(len(csv_data[i])):
                        float_data = float(csv_data[i][j])
                        if j == 0:
                            new_actor_x_list.append(float_data)
                        if j == 1:
                            new_actor_y_list.append(float_data)
            # actor trajectory
            if "new_1_2_actor_trajectory" in input_file_name:
                print(444)
                new_actor_trajectory_x_list = []
                new_actor_trajectory_y_list = []
                for i in range(len(csv_data)):
                    if i%20 == 0:
                        for j in range(len(csv_data[i])):
                            float_data = float(csv_data[i][j])
                            if j == 0:
                                new_actor_trajectory_x_list.append(float_data)
                            if j == 1:
                                new_actor_trajectory_y_list.append(float_data) 
            # predicted actor trajectory
            if "new_1_2_actor_pose" in input_file_name:
                print(555)
                new_actor_x_list = []
                new_actor_y_list = []
                for i in range(len(csv_data)):
                    for j in range(len(csv_data[i])):
                        float_data = float(csv_data[i][j])
                        if j == 0:
                            new_actor_x_list.append(float_data)
                        if j == 1:
                            new_actor_y_list.append(float_data) 
                actor_num = int(d_num)
                actor_vel_x = calc_difference(new_actor_x_list, actor_num)
                actor_vel_y = calc_difference(new_actor_y_list, actor_num)
                predicted_x_list = predicted_pose(new_actor_x_list, actor_vel_x)
                predicted_y_list = predicted_pose(new_actor_y_list, actor_vel_y)

        # plot
        plt.scatter(old_agent_x_list, old_agent_y_list, s=200, label="existing agent trajectory", color="green")
        plt.scatter(old_agent_trajectory_x_list, old_agent_trajectory_y_list, s=200, color="green", alpha=0.2)
        plt.scatter(old_actor_x_list, old_actor_y_list, s=200, label="existing actor trajectory", color="maroon")
        plt.scatter(old_actor_trajectory_x_list, old_actor_trajectory_y_list, s=200, color="maroon", alpha=0.2)

        plt.scatter(new_agent_x_list, new_agent_y_list, s=200, label="proposal agent trajectory", color="blue")
        plt.scatter(new_agent_trajectory_x_list, new_agent_trajectory_y_list, s=200, color="blue", alpha=0.2)
        plt.scatter(new_actor_x_list, new_actor_y_list, s=200, label="proposal actor trajectory", color="red")
        plt.scatter(new_actor_trajectory_x_list, new_actor_trajectory_y_list, s=200, color="red", alpha=0.2)
        plt.scatter(predicted_x_list, predicted_y_list, s=200, label="predicted pose", color="fuchsia", alpha=0.6)
        
        # plt.scatter(old_agent_x_list, old_agent_y_list, s=200, label="existing agent pose", color="darkblue")
        # plt.scatter(old_agent_trajectory_x_list, old_agent_trajectory_y_list, s=200, label="existing agent trajectory", color="darkblue", alpha=0.2)
        # plt.scatter(old_actor_x_list, old_actor_y_list, s=200, label="existing actor pose", color="maroon")
        # plt.scatter(old_actor_trajectory_x_list, old_actor_trajectory_y_list, s=200, label="existing actor_trajectory", color="maroon", alpha=0.2)
        # plt.scatter(new_agent_x_list, new_agent_y_list, s=200, label="proposal agent pose", color="blue")
        # plt.scatter(new_agent_trajectory_x_list, new_agent_trajectory_y_list, s=200, label="proposal agent trajectory", color="blue", alpha=0.2)
        # plt.scatter(new_actor_x_list, new_actor_y_list, s=200, label="proposal actor pose", color="red")
        # plt.scatter(new_actor_trajectory_x_list, new_actor_trajectory_y_list, s=200, label="proposal actor_trajectory", color="red", alpha=0.2)
        # plt.scatter(predicted_x_list, predicted_y_list, s=200, label="predicted pose", color="green", alpha=0.6)

        # specify ranges, labels, and grids
        plt.xlim(0, plot_size)
        plt.ylim(0, plot_size)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend()
        plt.grid(True)
        plt.show()
