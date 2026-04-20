import pandas as pd

df = pd.read_csv('../dataset/Master/products.csv')

# Gross profit margin = (price - cogs) / price
df['margin'] = (df['price'] - df['cogs']) / df['price']

# Average margin per segment
margin_by_segment = df.groupby('segment')['margin'].mean()

print(margin_by_segment.sort_values(ascending=False))
