import pandas as pd

df = pd.read_csv('../dataset/Transaction/orders.csv')
df['order_date'] = pd.to_datetime(df['order_date'])
# Drop duplicates for the same customer on the same day (if any)
df = df.drop_duplicates(subset=['customer_id', 'order_date'])
df = df.sort_values(by=['customer_id', 'order_date'])

counts = df['customer_id'].value_counts()
df = df[df['customer_id'].isin(counts[counts > 1].index)]

df['gap'] = df.groupby('customer_id')['order_date'].diff().dt.days
print("Median gap after dropping same-day orders:", df['gap'].median())
