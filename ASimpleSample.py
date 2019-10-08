import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
path = '/Users/Q-Q/Downloads/1567094400.android.txt'
df = pd.read_csv(path, sep=' ', header=None)
df.columns = ['timestamp', 'steps']
df['new_steps'] = df['steps'] - df.iloc[0, 1]
df['date'] = pd.to_datetime(df['timestamp'], unit='s') + pd.to_timedelta(8, 'h')
# df['weekday'] = df['date'].dt.dayofweek
df['minutes'] = df['date'].dt.minute + df['date'].dt.hour*60
df.drop_duplicates(['minutes'], keep='first', inplace=True)
df['Detal_time'] = df['minutes'].diff()
df['Detal_steps'] = df['steps'].diff()
df['freq'] = df['Detal_steps']/df['Detal_time']
print(df)
df.to_csv("android_sample.csv")
