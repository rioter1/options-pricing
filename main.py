import numpy as np   
import pandas as pd 
import requests  
import xlsxwriter 
import math 
import sklearn
stocks = pd.read_csv('sp_500_stocks.csv')
stocks.head(10)

IEX_CLOUD_API_TOKEN = 'Tpk_059b97af715d417d9f49f50b51b1c448'

symbol = 'AAPL'
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote/?token={IEX_CLOUD_API_TOKEN}'
#print(api_url)
data = requests.get(api_url).json()
#print(data['symbol'])
print(data)

price = data['latestPrice']
market_cap = data['marketCap']

my_columns = ['Ticker', 'Stock Price', 'Market Capitalization', 'Number of Shares to Buy']
final_dataframe = pd.DataFrame(columns = my_columns)


final_dataframe.append(
    pd.Series(
    [
        symbol,
        price,
        market_cap/1000000000,
        'N/A'
    ],
        index = my_columns),
    
    ignore_index=True
)


final_dataframe = pd.DataFrame(columns = my_columns)

for stock in stocks['Ticker'][:5]:
    api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote/?token={IEX_CLOUD_API_TOKEN}'
#print(api_url)
    data = requests.get(api_url).json()
    final_dataframe = final_dataframe.append(
    pd.Series(
    [
        stock,
        data['latestPrice'],
        data['marketCap'],
        'N/A'
    ],
    index = my_columns),
    ignore_index = True)


# we need to find a way to split our lists of tickers into sublists of list 100
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

symbol_groups = list(chunks(stocks['Ticker'], 100))


symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []  # need to transform all stocks that are into the list into a string, string will be passed in to the URL of the HTTP request
for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))
    
final_dataframe = pd.DataFrame(columns = my_columns)

for symbol_string in symbol_strings:
   # print(symbol_string)
    batch_api_call_url =f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote&token={IEX_CLOUD_API_TOKEN}'
    #print(batch_api_call_url)
    data = requests.get(batch_api_call_url).json()
    #print(data.status_code)
    
    for symbol in symbol_string.split(','):
        final_dataframe = final_dataframe.append(
        pd.Series(
        [
            symbol,
            data[symbol]['quote']['latestPrice'],
            data[symbol]['quote']['marketCap'],
            'N/A'
            
        ],
            index = my_columns),
            ignore_index = True
    )


portfolio_size = input('Enter the value of your portfolio:')

try:
    val = float(portfolio_size)
    
except ValueError:
    print("That's not a number! \nPlease try again:")
    portfolio_size = input('Enter the value of your portfolio:')
    val = float(portfolio_size)


position_size = val/len(final_dataframe.index)
num_apple_shares = position_size/500
print(math.floor(num_apple_shares))

position_size = val / len(final_dataframe.index)
for i in range(0, len(final_dataframe.index)):
    final_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size/final_dataframe.loc[i, 'Stock Price'])


# num_apple_shares = position_size/500
# print(math.floor(num_apple_shares))



df = dataset= final_dataframe

# applying k-means clustering to dataset
# Convert DataFrame to matrix
mat = dataset.values
# Using sklearn
km = sklearn.cluster.KMeans(n_clusters=5)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_
# Format results as a DataFrame
results = pd.DataFrame([dataset.index,labels]).T


# applying PCA to the dataframe 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plot


# You must normalize the data before applying the fit method
df_normalized=(df - df.mean()) / df.std()
pca = PCA(n_components=df.shape[1])
pca.fit(df_normalized)

# Reformat and view results
loadings = pandas.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
index=df.columns)
print(loadings)

plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()