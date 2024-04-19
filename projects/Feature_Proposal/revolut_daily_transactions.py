# -*- coding: utf-8 -*-
"""Revolut-Daily Transactions

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1taiRGEzf626V4-f5LscriBX8HKlSOQwi

Daily Transactions
"""

import numpy as np
import pandas as pd

import sys
!{sys.executable} -m pip install pandas

daily_transactions = pd.read_csv('Daily Household Transactions 5.csv')

daily_transactions.head()

daily_transactions.info()

daily_transactions.describe()

#Remove Currency Column
daily_transactions = daily_transactions.drop(["Currency"], axis=1)
daily_transactions.head()

#Rename 'Amount' column to add currency (INR)
daily_transactions.rename(columns={"Amount": "Amount (INR)"}, inplace=True)
daily_transactions.head()

#Sort transactions by date and remove timestamps
daily_transactions['Date'] = pd.to_datetime(daily_transactions['Date'], format='mixed', dayfirst=True).dt.normalize()
daily_transactions.sort_values('Date', inplace=True)
daily_transactions.head(2600)

#Renaming and compacting categories
category_type = daily_transactions['Category'].unique()
print(category_type)

#rename categories to match revolut spending categories
daily_transactions['Category'] = daily_transactions['Category'].replace({'Transportation': 'Transport',
                                                                         'Food': 'Groceries',
                                                                         'Culture': 'Outings',
                                                                         'Apparel': 'Shopping',
                                                                         'Household': 'Groceries',
                                                                         'Cook': 'Household Upkeep',
                                                                         'Rent': 'Utilities',
                                                                         'Public Provident Fund': 'Investment',
                                                                         'Money transfer': 'Transfers',
                                                                         'maid': 'Household Upkeep',
                                                                         'Gift': 'Gift',
                                                                         'Other': 'General',
                                                                         'Beauty': 'Beauty',
                                                                         'Health': 'Health',
                                                                         'Salary': 'Salary',
                                                                         'water (jar /tanker)': 'Groceries',
                                                                         'garbage disposal': 'Utilities',
                                                                         'Investment': 'Investment',
                                                                         'Recurring Deposit': 'Deposit',
                                                                         'Bonus': 'Bonus',
                                                                         'Family': 'Remittances/Gifts',
                                                                         'subscription': 'Subscription',
                                                                         'Festivals': 'Outings/Eating Out',
                                                                         'Grooming': 'Beauty',
                                                                         'Interest': 'Interest',
                                                                         'Education': 'Education',
                                                                         'Saving Bank account 2': 'Savings',
                                                                         'Gpay Reward': 'Interest',
                                                                         'Saving Bank account 1': 'Savings',
                                                                         'Life Insurance': 'Insurance',
                                                                         'Maturity amount': 'Investment Earnings',
                                                                         'Equity Mutual Fund A': 'Investment',
                                                                         'Equity Mutual Fund F': 'Investment',
                                                                         'Equity Mutual Fund E': 'Investment',
                                                                         'Fixed Deposit': 'Savings',
                                                                         'Equity Mutual Fund C': 'Investment',
                                                                         'Tourism': 'Travel',
                                                                         'Tax refund': 'Tax Refund',
                                                                         'Share Market': 'Investment',
                                                                         'Petty cash': 'Pocket Money',
                                                                         'Equity Mutual Fund D': 'Investment',
                                                                         'Self-development': 'General',
                                                                         'scrap': 'Cash',
                                                                         'Equity Mutual Fund B': 'Investment',
                                                                         'Small cap fund 1': 'Investment',
                                                                         'Small Cap fund 2': 'Investment',
                                                                         'Social Life': 'Outings',
                                                                         'Amazon pay cashback': 'Cashback',
                                                                         'Dividend earned on shares': 'Dividends',
                                                                         'Documents': 'General'})

daily_transactions['Category'].unique()

#Data cleaning - Moving data from 'subscriptions' to 'utilities'
daily_transactions.loc[daily_transactions['Subcategory'] == 'Mobile Service Provider', 'Category'] = 'Utilities'
daily_transactions.loc[daily_transactions['Note'] == 'electricity bill', 'Category'] = 'Utilities'
daily_transactions.loc[daily_transactions['Subcategory'] == 'Tata Sky', 'Category'] = 'Utilities'
daily_transactions.loc[daily_transactions['Subcategory'] == 'Wifi Internet Service', 'Category'] = 'Utilities'
daily_transactions.loc[daily_transactions['Subcategory'] == 'Mahanagar Gas', 'Category'] = 'Utilities'

#Data cleaning - Moving categories
daily_transactions.loc[daily_transactions['Note'] == 'Finding next Job', 'Category'] = 'Shopping'
daily_transactions.loc[daily_transactions['Note'] == 'Gift from inlaws', 'Category'] = 'Pocket Money'
daily_transactions.loc[daily_transactions['Note'] == 'From Family', 'Category'] = 'Pocket Money'
daily_transactions.loc[daily_transactions['Note'] == 'Shopping at Alfa', 'Category'] = 'Shopping'
daily_transactions.loc[daily_transactions['Note'] == 'HBR 2 Months subscription', 'Category'] = 'Subscription'
daily_transactions.loc[daily_transactions['Note'] == 'Home Food Delivery', 'Category'] = 'Outings/Eating Out'
daily_transactions.loc[daily_transactions['Category'] == 'Investment', 'Category'] = 'Investment'

#Remove 'Mode', 'Subcategory' and 'Note' columns
daily_transactions = daily_transactions.drop(["Mode", "Subcategory", "Note"], axis=1)
daily_transactions.head()

"""Preparation for Data Visualisation & Regression**"""

#Create a sub-dataframe to filter 2017 and 2018 data
transactions_2017_2018 = daily_transactions[(daily_transactions['Date'].dt.year == 2017) | (daily_transactions['Date'].dt.year == 2018)]
transactions_2017_2018.head()

#Display monthly, expenses & transfer-outs
# Filter data for income, expenses, and transfer-outs
income_transactions = transactions_2017_2018[transactions_2017_2018['Income/Expense'] == 'Income']
expense_transactions = transactions_2017_2018[transactions_2017_2018['Income/Expense'] == 'Expense']
transfer_out_transactions = transactions_2017_2018[transactions_2017_2018['Income/Expense'] == 'Transfer-Out']

# Group data by month and calculate totals for income, expenses, and transfer-outs
total_income = income_transactions.groupby(income_transactions['Date'].dt.to_period('M'))['Amount (INR)'].sum().reset_index()
total_income.rename(columns={'Amount (INR)': 'Total Income'}, inplace=True)

total_expenses = expense_transactions.groupby(expense_transactions['Date'].dt.to_period('M'))['Amount (INR)'].sum().reset_index()
total_expenses.rename(columns={'Amount (INR)': 'Total Expenses'}, inplace=True)

total_transfer_out = transfer_out_transactions.groupby(transfer_out_transactions['Date'].dt.to_period('M'))['Amount (INR)'].sum().reset_index()
total_transfer_out.rename(columns={'Amount (INR)': 'Total Transfer-Out'}, inplace=True)

# Display the calculated totals for each category
print("Total Income:")
print(total_income)
print("Total Expenses:")
print(total_expenses)
print("Total Savings/Investment:")
print(total_transfer_out)

"""**Data Regression**"""

import matplotlib.pyplot as plt
import plotly.express as px

import sys
!{sys.executable} -m pip install "notebook>=5.3" "ipywidgets>=7.5"
!{sys.executable} -m pip install plotly==4.14.3

!{sys.executable} -m pip install --force-reinstall plotly==4.14.3

#scatterplot Monthly Income, Expenses and Monthly Savings
monthly_totals_2017_2018 = pd.DataFrame({
    'Monthly Savings/Investments': total_transfer_out['Total Transfer-Out'],
    'Monthly Income': total_income['Total Income'],
    'Monthly Expenses': total_expenses['Total Expenses']
})
fig = px.scatter(monthly_totals_2017_2018, x="Monthly Income", y="Monthly Savings/Investments", color = "Monthly Expenses", trendline="ols", title = "Relationship between Monthly Income, Expenses and Savings/Investments")
fig.show()

#scatterplot Monthly Income and Expenses
fig = px.scatter(monthly_totals_2017_2018, x="Monthly Income", y="Monthly Expenses", trendline="ols", title = "Relationship between Monthly Income and Monthly Expenses")
fig.show()
monthly_totals_2017_2018['Month'] = monthly_totals_2017_2018.index.strftime('%Y-%m')

fig = px.scatter(monthly_totals_2017_2018, x="Monthly Income", y="Monthly Expenses", trendline="ols", title="Relationship between Monthly Income and Monthly Expenses",
                 text='Month')  # Use 'Month' column for text annotations

fig.update_traces(textposition='top center')  # Adjust the position of the text annotations
fig.show()

"""**Descriptive Statistics**"""

monthly_totals_2017_2018.mean()

monthly_totals_2017_2018.median()

monthly_totals_2017_2018.quantile([.1, .25, .5, .75])

# Interquartile range

fig = px.box(monthly_totals_2017_2018, y="Monthly Expenses", title="Box Plot of Monthly Expenses by Monthly Income")
fig.show()

# Note:
# Upper and lower fences cordon off outliers from the bulk of data in a set.
# Fences are usually found with the following formulas:

# Upper fence = Q3 + (1.5 * IQR)
# Lower fence = Q1 — (1.5 * IQR).

#standard deviation
monthly_totals_2017_2018.std()

#correlation
correlation_matrix = monthly_totals_2017_2018.corr()
print(correlation_matrix)

"""**Data Visualisation**"""



#stacked bar graph - Monthly income, expenses and savings
monthly_totals = total_income.merge(total_expenses, on='Date', how='outer').merge(total_transfer_out, on='Date', how='outer')
monthly_totals.plot(kind='bar', x='Date', stacked=True, figsize=(10, 6))
plt.title('2017-2018 Monthly Total Income, Expenses, and Savings')
plt.xlabel('Month')
plt.ylabel('Amount (INR)')
plt.xticks(rotation=45)
plt.legend(['Income', 'Expenses', 'Savings'])
plt.tight_layout()
plt.show()

"""**Data Forecasting**"""

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.tsa.stattools as st

monthly_totals['income_change'] = monthly_totals['Total Income'].transform(lambda x: (x - x.shift(1)) / x.shift(1) * 100)
monthly_totals['savings_change'] = monthly_totals['Total Transfer-Out'].transform(lambda x: (x - x.shift(1)) / x.shift(1) * 100)
monthly_totals['expenses_change'] = monthly_totals['Total Expenses'].transform(lambda x: (x - x.shift(1)) / x.shift(1) * 100)


# Drop NaN values
monthly_totals = monthly_totals.dropna()
monthly_totals.head(100)

monthly_totals['Month_Year'] = monthly_totals['Date'].dt.strftime('%b %Y')
fig1 = px.line(monthly_totals, x='Month_Year', y='income_change', title='Average income change over time')

fig1.show()

monthly_totals['Month_Year'] = monthly_totals['Date'].dt.strftime('%b %Y')
fig2 = px.line(monthly_totals, x='Month_Year', y='expenses_change', title='Average expenses change over time')

fig2.show()

# Visualize data
dates_ = pd.date_range(start='2017-01-01', end='2018-09-30', freq='MS')
plt.figure(figsize=(10, 6))
plt.plot(dates_, monthly_totals_2017_2018['Monthly Expenses']) # FIX
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Daily Transactions - Expenses')

plt.show()

# Visualize data
dates_ = pd.date_range(start='2017-01-01', end='2018-09-30', freq='MS')
plt.figure(figsize=(10, 6))
plt.plot(dates_, monthly_totals_2017_2018['Monthly Income']) # FIX
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Daily Transactions - Income')

plt.show()