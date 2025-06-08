# import sympy as sp

# x = sp.symbols('x')
# f = x**2
# derivative = sp.diff(f,x)
# print(f"Derivative:\n{derivative}") 

# # calculus for ML

# definite_integral = sp.integrate(f,(x,0,2))
# indefinite_integral = sp.integrate(f,x)
# print("Definite integral", definite_integral)
# print("Indefinite integral", indefinite_integral) 

# import numpy as np

# # generate synthetic data 
# np.random.seed(42)
# X = 2 * np.random.rand(100,1)
# y= 4 + 3 * X + np.random.randn(100,1)

# # add bias term to X
# X_b = np.c_[np.ones((100,1)),X]

# #SGD Implementation

# def stochastic_gradient_descent(X,y,theta,learning_rate,n_epochs):
#     m= len(y)
#     for epoch in range(n_epochs):
#         for i in range(m):
#             random_index = np.random.randint(m)
#             xi= X[random_index:random_index+1]
#             yi= y[random_index:random_index+1]
#             gradients = 2 * xi.T @ (xi @ theta - yi)
#             theta -= learning_rate * gradients 
#     return theta

# # initialise paratemers
# theta = np.random.randn(2,1)
# learning_rate = 0.01
# n_epochs = 50

# # pergorm sgd
# theta_opt = stochastic_gradient_descent(X_b,y,theta,learning_rate,n_epochs)
# print("Optimised Parameters", theta_opt)


# import numpy as np
# import matplotlib.pyplot as plt

# # Generate synthetic data
# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

# # Add bias term to X
# X_b = np.c_[np.ones((100, 1)), X]

# # Stochastic Gradient Descent (SGD) with tracking
# def stochastic_gradient_descent(X, y, theta, learning_rate, n_epochs):
#     m = len(y)
#     theta_history = []  # Store theta updates
#     for epoch in range(n_epochs):
#         for i in range(m):
#             random_index = np.random.randint(m)
#             xi = X[random_index:random_index + 1]
#             yi = y[random_index:random_index + 1]
#             gradients = 2 * xi.T @ (xi @ theta - yi)
#             theta -= learning_rate * gradients
#             theta_history.append(theta.copy())  # Track updates

#     return theta, theta_history

# # Initialize parameters
# theta = np.random.randn(2, 1)
# learning_rate = 0.01
# n_epochs = 50

# # Perform SGD and track progress
# theta_opt, theta_history = stochastic_gradient_descent(X_b, y, theta, learning_rate, n_epochs)

# # Convert history to numpy array for plotting
# theta_history = np.array(theta_history)

# # Plot convergence
# plt.figure(figsize=(8, 5))
# plt.plot(theta_history[:, 0], label="Theta 0 (Intercept)")
# plt.plot(theta_history[:, 1], label="Theta 1 (Slope)")
# plt.xlabel("Iterations")
# plt.ylabel("Parameter Value")
# plt.title("Parameter Convergence Over Iterations")
# plt.legend()
# plt.show()


# PROBABILITY THEOYR AND DISTRIBUTUONS  

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm,binom,poisson

# #Gausian distru

# x = np.linspace(-4,4,100)
# y = norm.pdf(x,loc=0,scale=1)
# plt.plot(x,y,label="Gaussian")
# plt.title("Gaussian Distribution")

# # binomial distru

# n, p = 10,0.5
# x=np.arange(0,n+1)
# y= binom.pmf(x,n,p)
# plt.bar(x,y,label="binomial")
# plt.title("Binomial distribution")
# plt.show()

# # poisson distru

# lam = 3
# x= np.arange(0,10)
# y = poisson.pmf(x,lam)
# plt.bar(x,y,label="poisson")
# plt.title("Poisson distribution")
# plt.show()


# Statistics fundamentals 
