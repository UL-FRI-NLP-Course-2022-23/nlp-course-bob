import pandas as pd
import os

def load_data(path):
    data = pd.DataFrame()
    
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        data_new = pd.read_csv(filepath)
        
        if not data.empty:
            data = pd.concat([data, data_new])
        else:
            data = data_new
            
    data.to_csv('paraphrases_all.csv')
    
if __name__ == "__main__":
    load_data('initial_data')