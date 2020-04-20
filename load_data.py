# Load the dataset into a pandas dataframe.
import pandas as pd

def get_data(tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t', names=['context', 'input', 'output'])
    df = df[1:]
    outputs = df.output.values
    inputs = df.input.values
    contexts = df.context.values
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    return contexts, inputs, outputs
