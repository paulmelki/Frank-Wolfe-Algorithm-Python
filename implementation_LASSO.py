#%% First round of testing of our implementation:
# We fix the number of obseverations (n) and the number of parameters (p)
# and study the performance of the algorithm when only lambda (l) is changing.

# Array holding the tolerance levels we will be working with
epsilon = np.array([0.1, 0.01, 0.001, 0.0001])

# Pandas dataframe that will hold the data used to create the plot
dataToPlot = pd.DataFrame({
    "epsilon" : np.array([0.1, 0.01, 0.001, 0.0001]),
    "k1" : np.zeros(4),
    "k2" : np.zeros(4),
    "k3" : np.zeros(4)
})

# CASE 1: l = 50, n = 1000, p = 700
# Array that will hold the returned numbers of iterations for each level of 
# tolerance.
returnedK = np.zeros(4)
# Array that will hold the returned data (f(x))
data = [None] * 4
# Array that will hold the norm of the difference: ||f(x_k) - (f_(k-1))
diffx = [None] * 4
n = 1000  # number of observations
p = 700   # number of parameters
l = 50    # penalty parameter
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k1'] = \
      frankWolfeLASSO(A, y, l=50, tol=epsilon[i])


# CASE 2: l = 500, n = 1000, p = 700
returnedK = np.zeros(4)
data = [None] * 4
diffx = [None] * 4
n = 1000
p = 700
l = 500
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k2'] = \
        frankWolfeLASSO(A, y, l=l, tol=epsilon[i])


# CASE 3: l = 5000, n = 1000, p = 700
returnedK = np.zeros(4)
data = [None] * 4
diffx = [None] * 4
n = 1000
p = 700
l = 5000
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k3'] = \
        frankWolfeLASSO(A, y, l=l, tol=epsilon[i])

# Plot the results
plt.plot(dataToPlot.epsilon, dataToPlot.k1, marker="o", color="darkgreen",\
              label = "l = 50")
plt.plot(dataToPlot.epsilon, dataToPlot.k2, marker="o", color="deepskyblue", \
              label = "l = 500")
plt.plot(dataToPlot.epsilon, dataToPlot.k3, marker="o", color="firebrick", \
              label = "l = 5000")
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Solving LASSO problem using Frank-Wolfe Algorithm", fontsize=16)
plt.xlabel("Tolerance level")
plt.ylabel("Number of iterations (k)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Question6b_toleranceIterations_l.png")
plt.show()



#%% Second round of testing of our implementation:
# We fix the number of obseverations (n) and lambda (l)
# and study the performance of the algorithm when only p is changing.

# Array holding the tolerance levels we will be working with
epsilon = np.array([0.1, 0.01, 0.001, 0.0001])

# Pandas dataframe that will hold the data used to create the plot
dataToPlot = pd.DataFrame({
    "epsilon" : np.array([0.1, 0.01, 0.001, 0.0001]),
    "k1" : np.zeros(4),
    "k2" : np.zeros(4),
    "k3" : np.zeros(4)
})

# CASE 1: n = 1000, p = 700, l = 50
# Array that will hold the returned numbers of iterations for each level of 
# tolerance.
returnedK = np.zeros(4)
# Array that will hold the returned data (f(x))
data = [None] * 4
# Array that will hold the norm of the difference: ||f(x_k) - (f_(k-1))
diffx = [None] * 4
n = 1000  # number of observations
p = 700   # number of parameters
l = 50    # penalty parameter
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k1'] = \
      frankWolfeLASSO(A, y, l=50, tol=epsilon[i])


# CASE 2: n = 1000, p = 1400, l = 50
n = 1000
p = 1400
l = 50
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k2'] = \
        frankWolfeLASSO(A, y, l=50, tol=epsilon[i])


# CASE 3: n = 1000, p = 7000, l = 50
n = 1000
p = 7000
l = 50
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k3'] = \
        frankWolfeLASSO(A, y, l=50, tol=epsilon[i])

# Plot the results
plt.plot(dataToPlot.epsilon, dataToPlot.k1, marker = "o", color = "darkslategray",\
              label = "n = 1000, p = 700")
plt.plot(dataToPlot.epsilon, dataToPlot.k2, marker = "o", color = "chartreuse", \
              label = "n = 1000, p = 1400")
plt.plot(dataToPlot.epsilon, dataToPlot.k3, marker = "o", color = "slateblue", \
              label = "n = 1000, p = 7000")
plt.legend()
plt.suptitle("Solving LASSO problem using Frank-Wolfe Algorithm", fontsize=16)
plt.xlabel("Tolerance level")
plt.ylabel("Number of iterations (k)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Question6b_toleranceIterations_p.png")
plt.show()



#%% Third round of testing of our implementation:
# We fix the number of parameters (p) and lambda (l)
# and study the performance of the algorithm when only n is changing.

# Array holding the tolerance levels we will be working with
epsilon = np.array([0.1, 0.01, 0.001, 0.0001])

# Pandas dataframe that will hold the data used to create the plot
dataToPlot = pd.DataFrame({
    "epsilon" : np.array([0.1, 0.01, 0.001, 0.0001]),
    "k1" : np.zeros(4),
    "k2" : np.zeros(4),
    "k3" : np.zeros(4)
})

# CASE 1: n = 1000, p = 1400, l = 50
# Array that will hold the returned numbers of iterations for each level of 
# tolerance.
returnedK = np.zeros(4)
# Array that will hold the returned data (f(x))
data = [None] * 4
# Array that will hold the norm of the difference: ||f(x_k) - (f_(k-1))
diffx = [None] * 4
n = 1000  # number of observations
p = 1400   # number of parameters
l = 50    # penalty parameter
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k1'] = \
      frankWolfeLASSO(A, y, l=50, tol=epsilon[i])


# CASE 2: n = 5000, p = 1400, l = 50
n = 5000
p = 1400
l = 50
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k2'] = \
        frankWolfeLASSO(A, y, l=50, tol=epsilon[i])


# CASE 3: n = 10000, p = 1400, l = 50
n = 10000
p = 1400
l = 50
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k3'] = \
        frankWolfeLASSO(A, y, l=50, tol=epsilon[i])

# Plot the results
plt.plot(dataToPlot.epsilon, dataToPlot.k1, marker = "o", color = "cadetblue",\
              label = "n = 1000, p = 1400")
plt.plot(dataToPlot.epsilon, dataToPlot.k2, marker = "o", color = "darkorange", \
              label = "n = 5000, p = 1400")
plt.plot(dataToPlot.epsilon, dataToPlot.k3, marker = "o", color = "midnightblue", \
              label = "n = 10000, p = 1400")
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Solving LASSO problem using Frank-Wolfe Algorithm", fontsize=16)
plt.xlabel("Tolerance level")
plt.ylabel("Number of iterations (k)")
plt.savefig("Question6b_toleranceIterations_n.png")
plt.show()
