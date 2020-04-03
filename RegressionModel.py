import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

data = pd.read_csv("data/Advertising.csv")

Xs = data.drop(['sales', 'Unnamed: 0'], axis=1)
y = data['sales'].values.reshape(-1, 1)

reg = LinearRegression()
reg.fit(Xs, y)
print("The linear model is: Y = {:.5} + {:.5}*TV + "
      "{:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0],
                                             reg.coef_[0][0],
                                             reg.coef_[0][1],
                                             reg.coef_[0][2]))

X = np.column_stack((data['TV'], data['radio'], data['newspaper']))
y = data['sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
