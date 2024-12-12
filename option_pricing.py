import numpy as np
from scipy.stats import norm


def option_price(S,K,T, r, sigma, type = "call"):

    d1 = (np.log(S/K) + ( r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if type == "call":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif type == "put":
        price = -S*norm.cdf(-d1) + K*np.exp(-r*T)*norm.cdf(d2)
    else:
        print("Error, the option is not call or put")

    return price
def delta(S,K,T, r, sigma, type = "call"):
    d1 = (np.log(S/K) + ( r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    if type == "call":
        delta = norm.cdf(d1)
    elif type == "put":
        delta = norm.cdf(d1) - 1
    else:
        print("Error, the option is not call or put")
    return delta
print("Value option ", option_price(40, 40, 1, 0.05, 0.22, "call"))  # 10.450583572185565
print("Delta value ", delta(40, 40, 1, 0.05, 0.22, "call"))  # 0.6320443415132453

# from scipy.optimize import brentq

# def implited_vol(price, S,K,T, r, type = "call"):
#     func = lambda sigma: option_price(S,K,T, r, sigma, type) - price
#     impl_vol = brentq(func, 1e-6, 10)
#     return impl_vol 

# S = 100  # Current stock price
# K_values = np.linspace(80, 120, 10)  # Strike prices
# T_values = np.linspace(0.1, 2, 5)  # Maturities
# r = 0.05  # Risk-free rate
# sigma_actual = 0.2  # Actual volatility for generating option prices
# option_type = "call"

# implied_vols = np.zeros((len(T_values), len(K_values)))
# for i, T in enumerate(T_values):
#     for j, K in enumerate(K_values):
#         market_price = option_price(S, K, T, r, sigma_actual, option_type)
#         implied_vols[i, j] = implited_vol(market_price, S, K, T, r, option_type)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# K_grid, T_grid = np.meshgrid(K_values, T_values)

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection="3d")

# # Surface plot
# ax.plot_surface(K_grid, T_grid, implied_vols, cmap="viridis", edgecolor="k")
# ax.set_xlabel("Strike Price")
# ax.set_ylabel("Time to Maturity")
# ax.set_zlabel("Implied Volatility")
# ax.set_title("Implied Volatility Surface")
# plt.show()
