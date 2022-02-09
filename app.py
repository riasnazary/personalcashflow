import pandas as pd
import numpy as np
import glob
import os.path
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import re
import datetime
import time

st.set_page_config(page_title="Finance Dashboard",
                  page_icon=":bar_chart:",
                  layout="wide"
                  )

df = pd.read_csv(
    '/Users/riasnazary/datascience/00_final-project/personalcashflow/data/transactionsfiltered/transactions_filtered.csv',
    ).iloc[:,1:]

# ---- SIDEBAR ----

st.sidebar.header("Please Filter Here:")
mcategory = st.sidebar.multiselect(
"Select the Major Category:",
    options=df["mcategory"].unique(),
    default=df["mcategory"].unique()
)

scategory = st.sidebar.multiselect(
"Select the Sub Category:",
    options=df["scategory"].unique(),
    default=df["scategory"].unique(),
)

auftraggeber = st.sidebar.multiselect(
"Select the Funder:",
    options=df["auftraggeber"].unique(),
    default=df["auftraggeber"].unique()
)

df_selection_exp = df.query(
    "betrag_eur < 0 & mcategory ==@mcategory & scategory ==@scategory & auftraggeber ==@auftraggeber"
)

df_selection_rev = df.query(
    "betrag_eur > 0 & mcategory ==@mcategory & scategory ==@scategory & auftraggeber ==@auftraggeber"
)

# ---- MAINPAGE ----

st.title(":bar_chart: Finance Dashboard")
# st.markdown("##")

# TOP KPI's
total_expenditures = df_selection_exp['betrag_eur'].sum()
total_revenues = df_selection_rev['betrag_eur'].sum()
average_value_by_transaction = round(df_selection_exp['betrag_eur'].mean(),2)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Expenditures: "f"{total_expenditures} EUR")
with middle_column:
    st.subheader("Total Revenues: "f"{total_revenues} EUR")
with right_column:
    st.subheader("Avg Transaction Value: "f"{average_value_by_transaction} EUR")
    # st.markdown("""---""")

# EXPENDITURES BY CATEGORY [BAR CHART]

exp_by_mcategory = (
    df_selection_exp.groupby(by=["mcategory"]).sum()[["betrag_eur"]].sort_values(by="betrag_eur")
)
fig_mcategory_value = px.bar(
    exp_by_mcategory,
    x="betrag_eur",
    y=exp_by_mcategory.index,
    orientation="h",
    title="<b>Expenditures by Major Category</b>",
    color_discrete_sequence=["crimson"] * len(exp_by_mcategory),
    template="plotly_white",
)
fig_mcategory_value.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

exp_by_scategory = (
    df_selection_exp.groupby(by=["scategory"]).sum()[["betrag_eur"]].sort_values(by="betrag_eur")
)
fig_scategory_value = px.bar(
    exp_by_scategory,
    x="betrag_eur",
    y=exp_by_scategory.index,
    orientation="h",
    title="<b>Expenditures by Sub Category</b>",
    color_discrete_sequence=["crimson"] * len(exp_by_scategory),
    template="plotly_white",
)
fig_scategory_value.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_mcategory_value, use_container_width=True)
right_column.plotly_chart(fig_scategory_value, use_container_width=True)

# --------------------------- MY NOTEBOOK ---------------------------
# --------------------------- MY NOTEBOOK ---------------------------
# --------------------------- MY NOTEBOOK ---------------------------
import pandas as pd
import numpy as np
import glob
import os.path
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

# read the latest csv-file of bank statements
main_path = r'/Users/riasnazary/datascience/00_final-project/personalcashflow/'
folder_path = main_path + 'data/giro'
file_type = '/*csv'
files = glob.glob(folder_path + file_type)
max_file = max(files, key=os.path.getctime)

# spliting bank statements in 3 parts: meta_data, standing_order and transactions
meta_data = pd.read_csv(max_file, sep = ";", encoding= 'unicode_escape',
                    skiprows = 1,
                    nrows = 5,
                    index_col = False,
                    names = ['designation','value']
                   )
transactions = pd.read_csv(max_file, sep = ";", encoding= 'unicode_escape', header = 18,
                          names = ['buchungsdatum','wertstellung','umsatzart','buchungsdetails',
                                     'auftraggeber','empfaenger','betrag_eur','saldo','NaN']
                          ).iloc[:,:8]

account_balance = meta_data.loc[meta_data['designation'] == 'Aktueller Kontostand', 'value'].values[0][:-5]

# transactions
    
df['buchungsdatum'] = pd.to_datetime(df['buchungsdatum'], errors='coerce')
revenues = df.query('betrag_eur > 0')
expenditures = df.query('betrag_eur < 0')
# expenditures are easier to grab when ratios to revenues are build
from datetime import datetime
from dateutil.relativedelta import relativedelta
df['buchungsdatum'] = pd.to_datetime(df['buchungsdatum'], errors='coerce')

# --------------------------- PREP: REVENUES ---------------------------
# --------------------------- PREP: REVENUES ---------------------------
# --------------------------- PREP: REVENUES ---------------------------

revenues = df.query('betrag_eur > 0')
mon_rev = (
revenues
    .assign(year = revenues['buchungsdatum'].dt.strftime('%Y'), 
           month = revenues['buchungsdatum'].dt.strftime('%b'))
    .groupby(['year','month'])
    .agg({'betrag_eur':'sum'})
    .sort_values(['year'], ascending=False)
)
# Define average revenue
avg_revenue = round(mon_rev.mean()[0],2)
# --------------------------- PREP: EXPENDITURES ---------------------------
# --------------------------- PREP: EXPENDITURES ---------------------------
# --------------------------- PREP: EXPENDITURES ---------------------------
expenditures = df.query('betrag_eur < 0')
mon_exp = (
expenditures
    .assign(year = expenditures['buchungsdatum'].dt.strftime('%Y'), 
           month = expenditures['buchungsdatum'].dt.strftime('%b'))
    .groupby(['year','month','scategory','mcategory'])
    .agg({'betrag_eur':'sum'})
    .sort_values(['year', 'month'], ascending=False)
)
b = []
def rate_builder(a):
    for i in range(len(a)):
        result = {}
        result['rate'] = round((abs(a[['betrag_eur']].iat[i,0])/avg_revenue),2)
        result['restrate'] = round((1-(abs(a[['betrag_eur']].iat[i,0])/avg_revenue)),2)
        b.append(result)
    return b
temp = pd.DataFrame(rate_builder(mon_exp))
temp.set_index(mon_exp.index,inplace=True)
mon_exp = mon_exp.join(temp)
mon_expm = (
mon_exp
    .groupby(['year','month','mcategory'])
    .agg({'betrag_eur':'sum', 'rate':'sum'})
    .sort_values(['year', 'month'], ascending=False)
    .reset_index()
)
last_1m = datetime.now() - relativedelta(months=1)
last_2m = datetime.now() - relativedelta(months=2)
last_3m = datetime.now() - relativedelta(months=3)
last1m, last2m, last3m = format(last_1m, '%b'), format(last_2m, '%b'), format(last_3m, '%b')
# defining last month and last 3 moths as quarter
mnumbers = mon_expm.query(f'month == "{last1m}"')
qnumbers = mon_expm.query(f'month == "{last1m}" | month == "{last2m}" | month == "{last3m}"')
mnumberso = (
mnumbers
    .groupby(['mcategory'])
    .agg({'betrag_eur':'sum', 'rate':'sum'})
    .reset_index()
    .sort_values('betrag_eur')
)
# Since we count here numbers of last 90 days, we cant sum the rate, which is why we need to redefine it here
qnumberso = (
qnumbers
    .groupby(['mcategory'])
    .agg({'betrag_eur':'sum', 
          # 'rate':'sum'
         })
    .reset_index()
    .sort_values('betrag_eur')
    .assign(rate = lambda x: round((abs(x['betrag_eur'])/(3*avg_revenue)),2))
)
# --------------------------- POTENTIAL OF CONSUMPTION ---------------------------
# --------------------------- POTENTIAL OF CONSUMPTION ---------------------------
# --------------------------- POTENTIAL OF CONSUMPTION ---------------------------
import plotly.graph_objects as go

# filter cash part out
part_of_cash = (
mon_exp
    .query('mcategory == "cash"')
    .groupby(['year','month','mcategory'])
    .agg({'betrag_eur':'sum', 'rate':'sum'})
    .sort_values(['year', 'month'], ascending=False)
    .reset_index()
    .assign(filtered = lambda x: x['betrag_eur'])
)
part_of_cash = part_of_cash[['year', 'month', 'filtered']]
monthly_expenditures = (
mon_exp
    .groupby(['year','month'])
    .agg({'betrag_eur':'sum', 'rate':'sum'})
    .sort_values(['year', 'month'], ascending=False)
    .reset_index()
)
monthly_expenditures = (
monthly_expenditures
    .merge(part_of_cash)
    .assign(betrag_eur = lambda x: x['betrag_eur']-x['filtered'])
)
monthly_expenditures.drop(['filtered'], axis=1, inplace=True)
monthly_expenditures_list = round(monthly_expenditures['betrag_eur'],2).abs().tolist()
avg_revenue_list = [avg_revenue for i in range(len(monthly_expenditures_list))]
year_month = monthly_expenditures.assign(year_month = lambda x: x['year']+" "+x['month'])
months = year_month['year_month'].tolist()
months, rev_numbers, exp_numbers = months, avg_revenue_list, monthly_expenditures_list
exp_rev_percent_change = [100*(1-(rev_count - exp_count) / rev_count)
                        for rev_count, exp_count in zip(rev_numbers, exp_numbers)]

# pass the text to the second graph_object only
fig2 = go.Figure(data=[
    go.Bar(name='Potential', x=months, y=rev_numbers, marker_color='darkgreen'),
    go.Bar(name='Consumed', x=months, y=exp_numbers, marker_color='crimson', 
        text=[f"{percent_change:.0f}%" if percent_change < 0 
              else f"{percent_change:.0f}%"
              for percent_change in exp_rev_percent_change ],
        textposition='inside', textfont_size=16, textfont_color='white')
])

fig2.update_layout(barmode='group', autosize=True, width=600, height=400,
                   title="<b>Potential of consumption per month</b>",
                  )
st.plotly_chart(fig2, use_container_width=True)

# --------------------------- EXP CHARTS ---------------------------
# --------------------------- EXP CHARTS ---------------------------
# --------------------------- EXP CHARTS ---------------------------

# In order to show the top categories, we applied quantiles

# show top 5 categories expenditures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import plotly.express as px

mquant75, qquant75 = mnumberso['betrag_eur'].quantile(.75), qnumberso['betrag_eur'].quantile(.75)
element1, element2 = mnumberso.query('betrag_eur < @mquant75')[:5],qnumberso.query('betrag_eur < @qquant75')[:5]

colors=["salmon","yellow","purple","chocolate","orangered"]
labels1 = element1['mcategory']
data1 = element1['betrag_eur'].abs()
fig_pie_exp1 = go.Figure(
    go.Pie(
        hole = .5,
        labels = labels1,
        values = data1,
        textinfo = "value"    
    )
)

fig_pie_exp1.update_layout(
    title="<b>Monthly Expenditures</b>",
    autosize=True, width=350, height=350
)
fig_pie_exp1.update_traces(
    hoverinfo='label+percent', 
    textinfo='value', 
    textfont_size=14,
    marker=dict(colors=colors)
)

colors=["salmon","yellow","purple","chocolate","orangered"]
labels1 = element2['mcategory']
data1 = element2['betrag_eur'].abs()
fig_pie_exp2 = go.Figure(
    go.Pie(
        hole = .5,
        labels = labels1,
        values = data1,
        textinfo = "value"
    )
)

fig_pie_exp2.update_layout(
    title="<b>Quarterly Expenditures</b>",
    autosize=True, width=350, height=350
)

fig_pie_exp2.update_traces(
    hoverinfo='label+percent', 
    textinfo='value', 
    textfont_size=14,
    marker=dict(colors=colors)
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_pie_exp1, use_container_width=True)
right_column.plotly_chart(fig_pie_exp2, use_container_width=True)

# --------------------------- CONSUMPTION CHART ---------------------------
# --------------------------- CONSUMPTION CHART ---------------------------
# --------------------------- CONSUMPTION CHART ---------------------------

import matplotlib.pyplot as plt

dl = (
expenditures
    .query('mcategory == "dailylife"')
    .groupby(['scategory'])
    .agg({'betrag_eur':'sum'})
    .assign(monthly = lambda x: round(x['betrag_eur']/3,2))
    .assign(weekly = lambda x: round(x['monthly']/(52/12),2))
    .assign(daily = lambda x: round(x['weekly']/(7),2))
    .assign(srate = lambda x: round(x['betrag_eur']/x['betrag_eur'].sum(),2))
    .sort_values('srate', ascending=False)
    .reset_index()
)
mconsumtion = mnumberso['rate'].sum() - mnumberso.query('mcategory == "cash"')['rate'].sum()
qconsumtion = qnumberso['rate'].sum() - qnumberso.query('mcategory == "cash"')['rate'].sum()
colors = ('crimson', 'darkgreen')

labels1 = ('consumption','potential')
data1 = [mconsumtion, (1-mconsumtion)]
fig_cons1 = go.Figure(
    data=[
        go.Pie(
            labels=labels1,
            values=data1,
            hole = .5
        )
    ]
)

fig_cons1.update_layout(
    title="<b>Potential of monthly surplus revenue</b>",
    autosize=True, width=350, height=350
)

fig_cons1.update_traces(
    hoverinfo='label+percent', 
    textinfo='value', 
    textfont_size=18,
    marker=dict(colors=colors)
)


colors = ['crimson', 'darkgreen']
labels2 = ('consumption','potential')
data2 = [qconsumtion, (1-qconsumtion)]
fig_cons2 = go.Figure(
    data=[
        go.Pie(
            labels=labels2,
            values=data2,
            hole = .5
        )
    ]
)

fig_cons2.update_layout(
    title="<b>Potential of quarterly surplus revenue</b>",
    autosize=True, width=350, height=350
)
fig_cons2.update_traces(
    hoverinfo='label+percent', 
    textinfo='value', 
    textfont_size=18,
    marker=dict(colors=colors)
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_cons1, use_container_width=True)
right_column.plotly_chart(fig_cons2, use_container_width=True)
                        
# --------------------------- ASSETS ---------------------------
# --------------------------- ASSETS ---------------------------
# --------------------------- ASSETS ---------------------------
             
# import other bank data
import PyPDF2
import sys
import io
from bs4 import BeautifulSoup
import requests
from yahoo_fin import stock_info as si

# # grabing balance from Postbank call
# path = main_path + r'data/postbank/call'
# file_type = '/*csv'
# files = glob.glob(path + file_type)
# max_file = max(files, key=os.path.getctime)
# postbank_call = pd.read_csv(max_file, sep = ";", encoding = 'unicode_escape', skiprows = 1, nrows = 5,
#                     index_col = False, names = ['designation','value'])

# # grabing balance from Leaseplan out of a pdf-file 
# pdf = main_path + 'data/leaseplan/Transaktionsuebersicht.pdf'
# pdfFileObj = open(pdf, 'rb')
# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
# pageObj = pdfReader.getPage(0)
# old_stdout = sys.stdout
# sys.stdout = buffer = io.StringIO()
# pageObj.extractText()
# print(pageObj.extractText())
# sys.stdout = old_stdout 
# leaseplan_call = buffer.getvalue()

# # grabing balance from Consorsbank portfolio
# path = main_path + r'data/consorsbank/portfolio'
# file_type = '/*csv'
# files = glob.glob(path + file_type)
# max_file = max(files, key=os.path.getctime)
# consorsbank_portfolio = pd.read_csv(max_file, sep = ";", encoding= 'unicode_escape',skiprows = 3, nrows = 2, 
#                                     index_col = False)

# # grabing balance from Consorsbank call
# path = main_path + r'data/consorsbank/call'
# file_type = '/*xls'
# files = glob.glob(path + file_type)
# max_file = max(files, key=os.path.getctime)
# consorsbank_call = pd.read_excel(max_file, index_col = False)

# # grabing balance from Consorsbank clearing
# path = main_path + r'data/consorsbank/clearing'
# file_type = '/*xls'
# files = glob.glob(path + file_type)
# max_file = max(files, key=os.path.getctime)
# consorsbank_clearing = pd.read_excel(max_file, index_col = False)

# # grabing balance from DKB credit
# path = main_path + r'data/dkb'
# file_type = '/*csv'
# files = glob.glob(path + file_type)
# max_file = max(files, key=os.path.getctime)
# dkb_credit = pd.read_csv(max_file, sep = ";", encoding= 'unicode_escape', on_bad_lines='skip',
#                          names = ['1','2'], skiprows = 3, nrows = 1, index_col = False)

# dkb_credit = dkb_credit.assign(x = dkb_credit['2'].str[:-4])
# dkb_credit = pd.to_numeric(dkb_credit['x'])
# dkb_value = dkb_credit[0]

# # grabing balance from Barclays credit
# path = main_path + r'data/barclaysbank'
# file_type = '/*xlsx'
# files = glob.glob(path + file_type)
# max_file = max(files, key=os.path.getctime)
# barclaysbank_credit = pd.read_excel(max_file, names = ['1','2'], skiprows = 6, nrows = 1, index_col = False)

# Requested stocks and etfs from portfolio1
list_stocks1 = ['ALV.DE', 'ABEA.DE', 'APC.DE', 'BRYN.DE', 'EMWE.DE', 'WDP.DE', 
               'EXXT.DE', 'EXS1.DE', 'EUNL.DE', 'MSF.DE', '3V64.DE', 'VOW.DE',]

consorsbank_portf = []
for i in list_stocks1:
    result = round(si.get_live_price(f'{i}'),2)
    consorsbank_portf.append(result)

portfolio1 = pd.read_csv(main_path+'data/portfolio/portfolio1.csv', index_col=0)
portfolio1 = portfolio1.assign(acqsum = lambda x: x['pcs']*x['acq'])
portfolio1['ticker'], portfolio1['price'] = list_stocks1, consorsbank_portf
portfolio1 = portfolio1.assign(pricesum = lambda x: x['pcs']*x['price'], profit = lambda x: x['pricesum'] - x['acqsum'])
profit_portfolio1 = portfolio1.pricesum.sum() - portfolio1.query('ticker == "BRYN.DE"')[['pricesum']].iat[0,0]
profit_portfolio1 = round(profit_portfolio1,2)

# Requested stocks and etfs from portfolio2
list_stocks2 = ['22UA.DE', 'ADB.DE', 'MMM.DE', 'AMD.DE']
traderepublic_portf = []
for i in list_stocks2:
    result = round(si.get_live_price(f'{i}'),2)
    traderepublic_portf.append(result)

portfolio2 = pd.read_csv(main_path+'data/portfolio/portfolio2.csv', index_col=0)
portfolio2 = portfolio2.assign(acqsum = lambda x: x['pcs']*x['acq'])
portfolio2['ticker'], portfolio2['price'] = list_stocks2, traderepublic_portf
portfolio2 = portfolio2.assign(pricesum = lambda x: x['pcs']*x['price'], profit = lambda x: x['pricesum'] - x['acqsum'])
profit_portfolio2 = round(portfolio2.pricesum.sum(),2)

# grab current gold price by web scraping
url = "https://www.goldpreis.de"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
parent = soup.find(class_ = 'table')
children = parent.contents[1]
dictionary = {"weight":[], "priceDollar":[], "priceEuro":[]}

for i, child in enumerate(children):
        dictionary['weight'].append(child.contents[0].get_text(strip=True))
        dictionary['priceDollar'].append(child.contents[1].get_text(strip=True))
        dictionary['priceEuro'].append(child.contents[2].get_text(strip=True))

gold_price = pd.DataFrame.from_dict(dictionary)


# Converting
# converter = pd.DataFrame({'translation': [
#     'Alltag',
                                          # 'Ruecklage',
                                          # 'Urlaub',
                                          # 'Fonds',
                                          # 'Eltern',
                                          # 'Puffer',
                                          # 'Gold',
                                          # 'Auslandskonto'
                                         # ], 
                          # 'values':[
    # # Alltag
    # account_balance.split(' ')[0][:-3], 
    # # Ruecklage
    # postbank_call.loc[postbank_call['designation'] == 'Aktueller Kontostand', 'value'].values[0][:-2],
    # # Urlaub
    # barclaysbank_credit.iat[0,1],
    # Fonds
    # consorsbank_portfolio.iat[0,0], 
    # # Eltern
    # consorsbank_call.keys()[0].split('\n')[-1][29:-4],
    # # Puffer
    # consorsbank_clearing.keys()[0].split('\n')[-2][29:-4],
    # Gold
    # gold_price.iloc[0:1,2:3].iat[0,0][:-2],
    # # Auslandskonto
    # leaseplan_call.split('\n')[5:6][0][:-2]
                          # ]})

# converter = converter.assign(values = converter['values'].str.replace('\.','', regex=True))
# converter = converter.assign(values = converter['values'].str.replace('\,','.', regex=True))
# converter = converter.assign(values = converter['values'].str[:-2])
# converter['values'] = pd.to_numeric(converter['values'])

# function to count monthly growth rate without interests
import datetime 
today = datetime.date.today()
today = today.strftime("%Y-%m-%d")

from datetime import datetime
from dateutil import relativedelta

def mgrowthrate (start, amount): 
    result = (relativedelta.relativedelta(datetime.strptime(today, "%Y-%m-%d"),
                                        datetime.strptime(start, "%Y-%m-%d")).months +
            relativedelta.relativedelta(datetime.strptime(today, "%Y-%m-%d"),
                                        datetime.strptime(start, "%Y-%m-%d")).years * 12)*amount
    return result

building_savings_value = mgrowthrate(start = '2018-11-01', amount=250+40)
company_pension_value = mgrowthrate(start = '2019-12-01', amount=268)

myassets = pd.DataFrame({'institution': 
                       ['Postbank', 'Postbank', 'DKB', 'Barclays Bank', 'Trade Republic', 
                        'Consors Bank','Consors Bank','Consors Bank', 
                        'Allianz', 'Alte Leipziger', '-', 'Leaseplan Bank', '-'],
                       'account': 
                       ['Giro','Savings','Credit','Credit','Portfolio',
                        'Portfolio','Call','Clearing','Company pension','Building saving','Physical resources', 
                        'Call', 'Time deposit'],
                      'designation': 
                       ['EC card','Reserve','Credit card for grocery','Credit card for holiday',
                        'Portfolio for shares',
                        'Portfolio for fonds','Pension parents', 'temporary',
                        'Pension personal', 'Savings for operations', 'Gold','Foreign call account', 
                        'Time deposit at will'],
                       'translation': 
                       ['Alltag','Ruecklage','Einkauf','Urlaub','Aktien','Fonds','Eltern','Puffer',
                        'Privatrente', 'Haus', 'Gold','Auslandskonto', 'Festgeld']
                      })
myassets['value'] = np.nan



# assemble myassets
import matplotlib.pyplot as plt

# adding some time values fromt the monthly growth rate function
# since we purchased 100 g of 99.9% Au, we will calculate with 100 x of the price per g 
# myassets.at[2,'value'] = dkb_value
myassets.at[0,'value'] = 0
myassets.at[1,'value'] = 0
myassets.at[2,'value'] = 0
myassets.at[3,'value'] = 0
myassets.at[4,'value'] = profit_portfolio2
myassets.at[5,'value'] = profit_portfolio1
myassets.at[6,'value'] = 0
myassets.at[7,'value'] = 0
myassets.at[8,'value'] = company_pension_value
myassets.at[9,'value'] = building_savings_value
myassets.at[10,'value'] = 0
myassets.at[11,'value'] = 0
myassets.at[12,'value'] = 0

# myassets = (
# myassets
#     .merge(converter, on="translation", how="outer")
#     .assign(value = lambda x: np.where(x['value'].isnull(), x['values'], x['value']))
#     .drop(['values'], axis=1)
# )

myassets_compressed = (
myassets
    .query('value > 0')
    .groupby(['account'])
    .agg({'value':'sum'})
    .sort_values('value', ascending=False)
    .reset_index()
)

# assets pie chart

colors=["forestgreen","peru","saddlebrown","chocolate","orangered","yellowgreen",'sandybrown','olive','tan']
labels1 = myassets_compressed.account.tolist()
data1 = myassets_compressed.value.tolist()
fig_asset = go.Figure(
    go.Pie(
        hole = .5,
        labels = labels1,
        values = data1,
        textinfo = "value"
    )
)

fig_asset.update_layout(
    title="<b>Assets</b>",
    autosize=True, width=500, height=500
)

fig_asset.update_traces(
    hoverinfo='label+percent', 
    textinfo='percent', 
    textfont_size=14,
    marker=dict(colors=colors)
)
st.plotly_chart(fig_asset, use_container_width=True)


# --------------------------- EVALUATION PART ---------------------------
# --------------------------- EVALUATION PART ---------------------------
# --------------------------- EVALUATION PART ---------------------------


botton=2
dailylife_rate=.2375

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st

list_1 = [mnumberso, qnumberso]
list_2 = ['monthly', 'quarterly']

st.title('\n\n\n\n\nFinancial evaluation')

st.markdown("""
<style>
.big-font {
    font-size:14px !important;
}
</style>
""", unsafe_allow_html=True)

for i, item in enumerate(list_1):
    if (item['rate'].sum() - item.query('mcategory == "cash"')['rate'].sum()) > 1:
        st.write("\n=================================== CRITICAL WARNING ====================================\n",
            f"\nYour latest {list_2[i]} revenue consumption are above your average revenue by:", 
              "\n=======================================================================================",
              round((i-1),2), "%. \n\nYou need to change your future behaviour!\n\n\n")


# case 2: good rating        
    elif (item['rate'].sum() - item.query('mcategory == "cash"')['rate'].sum()) < .7:
        st.markdown("\n====================================== LOOKING GOOD ===================================\n", unsafe_allow_html=True)          
        st.write(f"Your latest {list_2[i]} financial solidness complies a rating of A: good\n")
        st.write("\n=======================================================================================\n")

# case 3: not bad rating           
    elif (item['rate'].sum() - item.query('mcategory == "cash"')['rate'].sum()) < .8:
        st.markdown("\n================================= ROOM FOR IMPROVEMENT ===============================\n",unsafe_allow_html=True)
        st.write(f"Your latest {list_2[i]} financial solidness complies a rating of B: room for improvement")
        st.write("\nMay be you need to make some cuts on daily life. let's have a closer look on it:\n")

# case 3a: not bad rating, above dailylife_rate          
        if item.query('mcategory == "dailylife"')[['rate']].iat[0,0] >= dailylife_rate:
            st.write("\nYour rate for expenditures on daily life is over the threshold of",dailylife_rate*100,"%.",
                  "\nYour expenditures for",dl[['scategory','srate']][:1].iat[0,0],
                  "have a share of",dl[['scategory','srate']][:1].iat[0,1]*100,"%. \nThat would be a start.")

# case 3b: not bad rating, under dailylife_rate   
        elif item.query('mcategory == "dailylife"')[['rate']].iat[0,0] < dailylife_rate:
            st.write("   Your expenses on daily life are under the threshold of",dailylife_rate*100,
            "%.\n\n   May be you should consider making cuts on insurances, donations or other positions.\n"
              "\n=======================================================================================\n",             
             )

# case 4: poor rating              
    elif (item['rate'].sum() - i.query('mcategory == "cash"')['rate'].sum()) < .9:
        st.write("\n=================================== CRITICAL WARNING ====================================\n",
            f"Your latest {list_2[i]} financial solidness complies a rating of C: poor",
           "\n=============================================================================================\n",
             "\nYou should seriously consider making cuts on daily life. let's have a closer look on it:\n.")

# case 4a: poor rating, above dailylife_rate         
        if i.query('mcategory == "dailylife"')[['rate']].iat[0,0] >= dailylife_rate:
            st.write("\n   Your rate for expenditures on daily life is over the threshold of",dailylife_rate*100,"%.",
                  "\n   Your expenditures for",dl[['scategory','srate']][:1].iat[0,0],
                  "have a share of",dl[['scategory','srate']][:1].iat[0,1]*100,"That would be a start."
                 )
# case 4b: poor rating, under dailylife_rate            
        elif item.query('mcategory == "dailylife"')[['rate']].iat[0,0] < dailylife_rate:
            st.write("   Your expenses on daily life are under the threshold of",dailylife_rate,
                  "%.\n\n   May be you should consider making cuts on insurances, donations or other positions.",
                    "\n=======================================================================================\n")
           
        
# --------------------------- PORTFOLIO OPTIMIZATION ---------------------------
# --------------------------- PORTFOLIO OPTIMIZATION ---------------------------
# --------------------------- PORTFOLIO OPTIMIZATION ---------------------------

from pandas_datareader import data as web
from datetime import datetime
import datetime
import matplotlib.pyplot as plt

# get the stocks/ portfolio starting date
past10year = (datetime.date.today() - datetime.timedelta (days=365*10)).strftime("%Y-%m-%d")
past05year = (datetime.date.today() - datetime.timedelta (days=365*5)).strftime("%Y-%m-%d")
past02year = (datetime.date.today() - datetime.timedelta (days=365*2)).strftime("%Y-%m-%d")
past01year = (datetime.date.today() - datetime.timedelta (days=365)).strftime("%Y-%m-%d")
pasthalfyear = (datetime.date.today() - datetime.timedelta (days=183)).strftime("%Y-%m-%d")
pastquarter = (datetime.date.today() - datetime.timedelta (days=90)).strftime("%Y-%m-%d")
pastmonth = (datetime.date.today() - datetime.timedelta (days=30)).strftime("%Y-%m-%d")
today = (datetime.date.today()).strftime("%Y-%m-%d")

# we take our upper defined list of stock names 
assets = list_stocks1 + list_stocks2
# weights
weights = np.array([(1/len(assets)) for i in range(len(assets))])
# create a df to store the adjusted close price of the stocks
df = pd.DataFrame()
for stock in assets:
    df[stock] = web.DataReader(stock, data_source = 'yahoo', start = past10year, end = today)['Adj Close']

st.title('\n\n\n\n\nPortfolio recommender')
st.write('According to your current portfolio setup, we have the following values: ')
    
returns = df.pct_change()
cov_matrix_annual = returns.cov() * 252
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_volatility = np.sqrt(port_variance)
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252

percent_var = str(round(port_variance,2)*100)
percent_vols = str(round(port_volatility,2)*100)
percent_ret = str(round(portfolioSimpleAnnualReturn,2)*100)

currentpf = pd.DataFrame(
    {
        'Desgination': ['Expected annual return', 'Annual volatility / risk', 'Annual variance'],
        'Current_Rates': [percent_var, percent_vols, percent_ret]
    }
)
st.dataframe(currentpf)

# portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# calculate the expected returns and the annualised sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# optimize for max sharpe ratio (from william sharpe, 1966)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

st.write('According to the efficient frontier tool, the follwing re-allocation will be recommended:')

rec_df1 = pd.DataFrame({'Name':cleaned_weights.keys(), 'Purposed_Allocation':cleaned_weights.values()})

# get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

pfval = 100000
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = pfval)
pfval = str(pfval)

allocation, leftover = da.lp_portfolio()
# st.write('Discrete allocation: ', allocation)
rec_df2 = pd.DataFrame({'Ticker':allocation.keys(), 'Purposed_Allocation':allocation.values()}).sort_values('Purposed_Allocation', ascending=False)
st.dataframe(rec_df2)

st.markdown('With a Total Portfolio Value of € *'+pfval+'* € your remaining Funds will be: € *{:.2f}*'.format(leftover))

potentialpf = pd.DataFrame(
    {
        'Desgination': ['Expected annual return', 'Annual volatility / risk', 'Annual variance'],
        'Potential_Rates': ef.portfolio_performance(verbose = False)
    }
)
potentialpf = potentialpf.assign(Potential_Rates = lambda x: round((x['Potential_Rates']*100),2))
st.dataframe(potentialpf)