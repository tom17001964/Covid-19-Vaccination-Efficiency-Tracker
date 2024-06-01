# main.py
# Author: Jack G. Robinson Heath
# GitHub: https://github.com/tom17001964

from data_processor.processor import DataProcessor
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Adjust the logging level as needed

# Example Usage
data_processor = DataProcessor("data.csv")
data_processor.optimize_columns()
data_processor.convert_date_to_datetime()
data_processor.extract_year()
data_processor.calculate_vaccination_efficiency()
data_processor.normalize_vaccination_efficiency()
data_processor.train_ridge_model(target_variable='vaccination_efficiency')
