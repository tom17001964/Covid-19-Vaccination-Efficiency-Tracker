import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from prettytable import PrettyTable
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a console handler and set the level to INFO
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and attach it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(console_handler)

        try:
            self.data = pd.read_csv(data_path)
            self.logger.info("Data loaded successfully.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {data_path}")
            raise FileNotFoundError(f"File not found: {data_path}")

        self.columns_to_optimize = ['population', 'total_vaccinations', 'stringency_index', 'positive_rate']
        self.years = [2021, 2022, 2023]

        # Initialize the new columns
        self.data['vaccination_efficiency'] = 0.0
        self.data['normalized_vaccination_efficiency'] = 0.0
        self.data['normalized_vaccination_efficiency_percentage'] = 0.0

    def optimize_columns(self):
        self.data[self.columns_to_optimize] = self.data[self.columns_to_optimize].apply(lambda x: np.nan_to_num(x, nan=1))
        self.logger.info("Columns optimized successfully.")

    def convert_date_to_datetime(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.logger.info("Date conversion to datetime successful.")

    def extract_year(self):
        self.data['year'] = self.data['date'].dt.year
        self.logger.info("Year extraction successful.")

    def calculate_vaccination_efficiency(self):
        self.data['vaccination_efficiency'] = (
            self.data['total_vaccinations'] / self.data['population']
        ) * (1 - (self.data['total_deaths'] / self.data['population'])) * self.data['stringency_index']
        self.logger.info("Vaccination efficiency calculation successful.")

    def normalize_vaccination_efficiency(self):
        for year in self.years:
            data_year = self.data[self.data['year'] == year]

            # Calculate min_value and max_value using .loc
            min_value = data_year['vaccination_efficiency'].min()
            max_value = data_year['vaccination_efficiency'].max()

            if max_value != min_value:  # To avoid division by zero
                # Calculate the normalized_vaccination_efficiency using .loc
                self.data.loc[self.data['year'] == year, 'normalized_vaccination_efficiency'] = (
                        data_year['vaccination_efficiency'] - min_value) / (max_value - min_value)

                # Calculate the normalized_vaccination_efficiency_percentage using .loc
                self.data.loc[self.data['year'] == year, 'normalized_vaccination_efficiency_percentage'] = self.data.loc[
                    self.data['year'] == year, 'normalized_vaccination_efficiency'] * 100
            else:
                self.data.loc[self.data['year'] == year, 'normalized_vaccination_efficiency'] = 0
                self.data.loc[self.data['year'] == year, 'normalized_vaccination_efficiency_percentage'] = 0

        self.logger.info("Normalization of vaccination efficiency successful.")

    def train_ridge_model(self, target_variable, alpha=2.0, test_size=0.3, random_state=42):
        try:
            # Extract features and target variable
            X = self.data[['normalized_vaccination_efficiency']]
            y = self.data[target_variable].fillna(0)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Initialize and train the Ridge regression model
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(X_train, y_train)

            # Predict and calculate mean squared error
            y_pred = ridge_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.logger.info(f'Mean Squared Error: {mse}')
            self.logger.info(f'R^2 Score: {r2}')

            # Display information using PrettyTable for both years
            table = PrettyTable()
            table.field_names = ["Sample (Day)", "Year", "Actual", "Predicted"]

            total_efficiency = {}
            
            # Use a single loop to process each year's data and predictions
            for year in [2021, 2022]:
                data_year = self.data[self.data['year'] == year]
                X_year = data_year[['normalized_vaccination_efficiency']]
                y_year = data_year[target_variable]

                y_pred_year = ridge_model.predict(X_year)

                # Use vectorized operations to round the predictions
                rounded_actuals = y_year.round(2).values
                rounded_predictions = y_pred_year.round(2)

                for i, (actual, predicted) in enumerate(zip(rounded_actuals, rounded_predictions)):
                    table.add_row([i + 1, year, actual, predicted])
                
                # Calculate total efficiency percentage for the year
                total_efficiency[year] = self.data.loc[self.data['year'] == year, 'normalized_vaccination_efficiency_percentage'].mean()
            
            print("\nRidge Regression Results:")
            print(table)
            
            # Print total efficiency percentage for each year
            for year, efficiency in total_efficiency.items():
                print(f"Total Efficiency Percentage for {year}: {efficiency:.2f}%")

            # Calculate compound growth rate of vaccination efficiency
            if 2021 in total_efficiency and 2022 in total_efficiency:
                efficiency_2021 = total_efficiency[2021]
                efficiency_2022 = total_efficiency[2022]
                if efficiency_2021 != 0:  # To avoid division by zero
                    compound_growth_rate = ((efficiency_2022 / efficiency_2021) ** (1 / 1)) - 1
                    print(f"Compound Efficiency Growth Rate from 2021 to 2022: {compound_growth_rate * 100:.2f}%")
                else:
                    print("Compound Efficiency Growth Rate cannot be calculated due to zero efficiency in 2021.")
            else:
                print("Either 2021 or 2022 (or both) not found in total_efficiency.")
        except Exception as e:
            self.logger.error(f"Error in train_ridge_model: {e}")
