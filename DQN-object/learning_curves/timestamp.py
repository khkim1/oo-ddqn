import os
import numpy as np





dir_name = '/Users/KunhoKim/Desktop/StanfordResearch/relDQN/DQN-object/log/2017-10-29_11-01-36_PongDeterministic-v4_True/'

time_array = []

print(os.listdir(dir_name))

for item in os.listdir(dir_name):
    time_array.append(os.path.getmtime(dir_name + item))

time_array = np.array(time_array)
time_array -= np.min(time_array)
time_array /= 3600.

t_sorted = np.sort(time_array)


np.savetxt('timestamps_object.txt', t_sorted)
