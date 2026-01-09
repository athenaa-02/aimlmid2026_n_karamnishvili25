# aimlmid2026_n_karamnishvili25
Midterm exam in AI and Machine Learning

# assignment 1, Finding the correlation  
The purpose of this analysis is to measure the strength and direction of the linear relationship between two numerical variables, x and y, using the Pearson correlation coefficient.


Step 1: Data Preparation

Two datasets were given:
x: independent variable
y: dependent variable


Step 2: Compute the Means

The arithmetic mean of each dataset is calculated:
x_mean = sum(x) / n
y_mean = sum(y) / n
The means represent the central value of each variable and are used as reference points for further calculations.


Step 3: Compute Deviations from the Mean

For each observation:
    dx = x[i] - x_mean
    dy = y[i] - y_mean
These deviations show how far each value is from its mean.


Step 4: Covariance Calculation

The covariance measures whether the variables move together:
  cov += dx * dy

Positive covariance : variables increase together
Negative covariance: one increases while the other decreases

In this case, the covariance is negative, indicating an inverse relationship.



Step 5: Standardization

To ensure the result is scale-independent, the covariance is divided by the product of the standard deviations:
sigma = math.sqrt(sum_x2 * sum_y2)
r = cov / sigma

Step 6: Pearson Correlation Coefficient
r =  -0.6554845380710971
r ≈ −0.65

The value of r lies between -1 and 1 and indicates a moderate negative linear correlation

As x increases, y tends to decrease

please view correlation visualisation image


# assignment 2, Spam email detection