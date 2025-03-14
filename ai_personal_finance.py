import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import requests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class ExpenseTracker:
    def __init__(self):
        self.expenses = pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Description'])
    
    def add_expense(self, date, category, amount, description):
        new_expense = {'Date': pd.to_datetime(date), 'Category': category, 'Amount': amount, 'Description': description}
        self.expenses = self.expenses.append(new_expense, ignore_index=True)
    
    def get_expenses(self):
        return self.expenses
    
    def get_expense_by_category(self, category):
        return self.expenses[self.expenses['Category'] == category]
    
    def get_monthly_summary(self):
        self.expenses['Month'] = self.expenses['Date'].dt.to_period('M')
        return self.expenses.groupby('Month').sum()['Amount']
    
class IncomeTracker:
    def __init__(self):
        self.income = pd.DataFrame(columns=['Date', 'Source', 'Amount', 'Description'])
    
    def add_income(self, date, source, amount, description):
        new_income = {'Date': pd.to_datetime(date), 'Source': source, 'Amount': amount, 'Description': description}
        self.income = self.income.append(new_income, ignore_index=True)
    
    def get_income(self):
        return self.income
    
    def get_monthly_income_summary(self):
        self.income['Month'] = self.income['Date'].dt.to_period('M')
        return self.income.groupby('Month').sum()['Amount']
    
class BudgetPlanner:
    def __init__(self):
        self.budgets = pd.DataFrame(columns=['Category', 'Budget_Amount', 'Actual_Spent'])
        
    def set_budget(self, category, budget_amount):
        self.budgets = self.budgets.append({'Category': category, 'Budget_Amount': budget_amount, 'Actual_Spent': 0}, ignore_index=True)
    
    def update_actual_spent(self, category, amount):
        self.budgets.loc[self.budgets['Category'] == category, 'Actual_Spent'] += amount
    
    def get_budget_summary(self):
        self.budgets['Remaining'] = self.budgets['Budget_Amount'] - self.budgets['Actual_Spent']
        return self.budgets
    
class FinancialPredictor:
    def __init__(self, income_data, expense_data):
        self.income_data = income_data
        self.expense_data = expense_data
    
    def prepare_data(self):
        self.income_data['Income'] = self.income_data['Amount']
        self.expense_data['Expense'] = self.expense_data['Amount']
        data = pd.merge(self.income_data, self.expense_data, on='Date', how='outer').fillna(0)
        data['Net_Cashflow'] = data['Income'] - data['Expense']
        return data
    
    def train_model(self, data):
        data['Date'] = pd.to_datetime(data['Date']).map(datetime.timestamp)
        X = data[['Date']].values
        y = data['Net_Cashflow'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        print(f'Model Mean Squared Error: {error}')
        
    def predict_future_cashflow(self, future_dates):
        future_dates = pd.to_datetime(future_dates).map(datetime.timestamp).values.reshape(-1, 1)
        predictions = self.model.predict(future_dates)
        return predictions

class PersonalFinanceApp:
    def __init__(self):
        self.expense_tracker = ExpenseTracker()
        self.income_tracker = IncomeTracker()
        self.budget_planner = BudgetPlanner()
        
    def add_expense(self, date, category, amount, description):
        self.expense_tracker.add_expense(date, category, amount, description)
        
    def add_income(self, date, source, amount, description):
        self.income_tracker.add_income(date, source, amount, description)
        
    def set_budget(self, category, budget_amount):
        self.budget_planner.set_budget(category, budget_amount)
    
    def update_budget(self, category, amount):
        self.budget_planner.update_actual_spent(category, amount)
    
    def get_expense_summary(self):
        return self.expense_tracker.get_monthly_summary()
    
    def get_income_summary(self):
        return self.income_tracker.get_monthly_income_summary()
    
    def get_budget_summary(self):
        return self.budget_planner.get_budget_summary()
    
    def predict_cashflow(self, income_data, expense_data, future_dates):
        predictor = FinancialPredictor(income_data, expense_data)
        predictor.prepare_data()
        predictor.train_model(predictor.prepare_data())
        predictions = predictor.predict_future_cashflow(future_dates)
        return predictions

if __name__ == '__main__':
    app = PersonalFinanceApp()
    
    app.add_income('2023-01-15', 'Job', 3000, 'Monthly Salary')
    app.add_expense('2023-01-17', 'Groceries', 200, 'Weekly grocery shopping')
    app.add_expense('2023-01-20', 'Utilities', 150, 'Monthly electricity bill')
    
    app.set_budget('Groceries', 800)
    app.update_budget('Groceries', 200)
    
    income_summary = app.get_income_summary()
    expense_summary = app.get_expense_summary()
    budget_summary = app.get_budget_summary()

    print('Income Summary:\n', income_summary)
    print('Expense Summary:\n', expense_summary)
    print('Budget Summary:\n', budget_summary)

    future_dates = ['2023-02-15', '2023-03-15', '2023-04-15']
    future_predictions = app.predict_cashflow(app.income_tracker.get_income(), app.expense_tracker.get_expenses(), future_dates)
    
    print('Future Cashflow Predictions:\n', future_predictions)
    
    plt.figure(figsize=(12,6))
    income_summary.plot(kind='bar', color='green', alpha=0.7, label='Income', position=0)
    expense_summary.plot(kind='bar', color='red', alpha=0.7, label='Expenses', position=1)
    plt.title('Monthly Income vs Expenses')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid()
    plt.show()