from scipy.stats import lognorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('output/12s_auctions.csv')

# Lognormális eloszlás paramétereinek becslése
shape, loc, scale = lognorm.fit(data['maximal_bid'], floc=0)

# Illesztett eloszlás megjelenítése
x = np.linspace(data['maximal_bid'].min(), data['maximal_bid'].max(), 1000)
lognorm_pdf = lognorm.pdf(x, shape, loc, scale)

# Ábrázolás
plt.figure(figsize=(10, 6))
plt.hist(data['maximal_bid'], bins=30, density=True, color='skyblue', alpha=0.6, label='Data Histogram')
plt.plot(x, lognorm_pdf, 'g-', lw=2, label='Lognormális Fit')
plt.title('Fit: Lognormális Eloszlás')
plt.xlabel('Maximal Bid')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

