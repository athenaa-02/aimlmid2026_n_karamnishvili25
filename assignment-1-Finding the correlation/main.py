import math

x = [1, 3, 5, 7, -1.5 , -5 ,-6.2, -9.5]
y = [-2, 1, -3, -2, 0.8, -1, 2, 1]


n = len(x)

x_mean = sum(x) / n
y_mean = sum(y) / n

cov = 0
sum_x2 = 0
sum_y2 = 0

for i in range(n):
    dx = x[i] - x_mean
    dy = y[i] - y_mean
    cov += dx * dy
    sum_x2 += dx ** 2
    sum_y2 += dy ** 2

sigma = math.sqrt(sum_x2 * sum_y2)

r = cov / sigma

print("pearson correlation coefficient:", r)
