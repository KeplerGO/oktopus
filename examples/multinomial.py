from pyMLE import MultinomialLikelihood
import numpy as np

T = 6*60
n_bins = 6*60
t = np.linspace(0, T, n_bins)
decay = 1
data = np.exp(-decay * t) + np.random.normal(loc=0.0, scale=0.05, size=n_bins)

def exp_model(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pov = curve_fit(exp_model, t, data)
plt.plot(t, exp_model(t, *popt), 'k--')
plt.scatter(t, data, s=2)

def exp_prob_model(r, bkg, n_bins):
    i = np.arange(1, n_bins+1)
    pi = (np.exp(-i * r / n_bins) * (np.exp(r / n_bins) - 1) /
          (1 - np.exp(-r)))
    return bkg / n_bins + (1 - bkg) * pi

model = lambda r, b: exp_prob_model(r, b, n_bins=n_bins)
logL = MultinomialLikelihood(data=data, prob_model=model)
params = logL.fit(x0=[10.0, 0.0], method='Nelder-Mead')

plt.plot(t, data.sum() * model(*params.x), 'r:')
plt.show()
