import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# import statsmodels.api as sm

data = pd.read_csv("data/Advertising.csv")

print(data.head())

