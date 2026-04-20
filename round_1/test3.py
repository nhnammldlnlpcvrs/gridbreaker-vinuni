import pandas as pd

df_returns = pd.read_csv('../dataset/Transaction/returns.csv')
df_products = pd.read_csv('../dataset/Master/products.csv')

# Join on product_id
df_joined = pd.merge(df_returns, df_products, on='product_id')

# Filter for Streetwear category
# Note: we need to verify the exact column name and value for Streetwear
df_streetwear = df_joined[df_joined['category'] == 'Streetwear']

# Count return_reasons
print(df_streetwear['return_reason'].value_counts())
