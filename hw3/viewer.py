import pandas as pd
from IPython.display import display

df = pd.read_pickle("Top10_music.pickle")
pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', -1)
display(df)