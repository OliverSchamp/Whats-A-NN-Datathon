import pandas as pd

histograms = pd.DataFrame({'artwork': pd.Series(dtype='str'), 'model': pd.Series(dtype='str'),
                         'histogram': pd.Series(dtype='int')})

histograms.to_csv('histograms.csv', index=False)