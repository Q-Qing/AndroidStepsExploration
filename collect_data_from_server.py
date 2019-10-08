import os
import re
import pandas as pd
from training_data_collection import create_csv_for_user

pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 50000)

"""
root_path = '/var/www/jdjbz/data/'
result_path = '/home/kqshi/detailed_steps/'

# list all folders of root_path
files = os.listdir(root_path)
for user_id in files:
    # check the folder whether is users' folder
    if re.match("^\d", user_id):
        print(user_id)
        path = os.path.join(root_path, user_id+"/steps")
        if os.path.exists(path):
            print(path+" exist")
            create_csv_for_user(path, result_path, user_id)
        else:
            print("no detailed steps")
"""

root_path = '/Users/Q-Q/data set/step dataset/847471356/steps'
user_id = "845471356"
result_path = "data/"
create_csv_for_user(root_path, result_path, user_id)