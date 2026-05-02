import pandas as pd

df = pd.read_csv('experiments/results.csv', header=None,
                 names=['run_id','episode','avg_reward',
                        'avg_waiting_time','epsilon','alpha','gamma'])

df.to_csv('experiments/results.csv', index=False)
print("Headers added to results.csv")
print(df.head())