import os
import pandas as pd

def load_csv(data_folder_path):
    data = pd.DataFrame()
    files = []
    counter = 0
    for root, dirs, files in os.walk(data_folder_path):
        for i,file in enumerate(files):
            if file.endswith(".csv"):
                f = os.path.join(root, file)
                counter += 1
                #print(f)
                exp = pd.read_csv(f)
                data = pd.concat( (data,  exp),sort = True )
    print('{} cvs files were loaded'.format(counter))
    return data