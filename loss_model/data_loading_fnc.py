import pandas as pd
import loss_model
# Load the data
def load_the_data(path, line_step = None, is_test: bool = False):
    if is_test==True:
        df = pd.read_csv(path, skiprows = lambda i: i % line_step != 0)
    else:
        df = pd.read_csv(path)
    return df