import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

# line smoothing
from scipy.interpolate import make_interp_spline, BSpline

# Assign the data to predictor and outcome variables
# Load the data
train_data = pd.read_csv('poly_reg_data.csv')

# setup: X predictor
X = train_data['Var_X'].values.reshape(23, 1)
#print(X.shape)

# dependent var
y = train_data['Var_Y']


# sort data for plot
inds = X.ravel().argsort()    # Here I am assuming that x has single feature     
x = X.ravel()[inds].reshape(-1,1)
y = y.values[inds]


# Create polynomial features
# Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree=8)
X_poly = poly_feat.fit_transform(x)

# Make and fit the polynomial regression model
# Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!

# Plot outputs
# draw the points 
plt.scatter(x, y)


# -- smoothing ----
#define x as 200 equally spaced values between the min and max of original x 
xnew = np.linspace(x.min(), x.max(), 200) 

#define spline
x_myarr = np.array(x.flatten())
#print(x_myarr.ndim)
#print(np.any(x_myarr[1:] <= x_myarr[:-1]))
y_myarr = np.array(y.flatten())
#spl = make_interp_spline(x_myarr, y_myarr, k=7)
spl = make_interp_spline(x_myarr, poly_model.predict(X_poly), k=3)

y_smooth = spl(xnew)

#plt.plot(xnew, poly_model.predict(X_poly), '-r')
plt.plot(xnew, y_smooth)
plt.show()
