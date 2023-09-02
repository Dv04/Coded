from Pearson import pearson
from Spearman import spearman
from Kendall import kendall

x = []
y = []


print ('Pearson Rho: %f' % pearson(x, y))

print ('Spearman Rho: %f' % spearman(x, y))

print ('Kendall Tau: %f' % kendall(x, y))