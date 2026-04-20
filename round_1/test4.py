import pandas as pd

df = pd.read_csv('../dataset/Operational/web_traffic.csv')
mean_bounce_rate = df.groupby('traffic_source')['bounce_rate'].mean().sort_values()
print(mean_bounce_rate)
