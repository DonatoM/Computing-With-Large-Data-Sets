import nltk
import pandas
import numpy
from scipy.stats import kurtosis
from scipy.spatial.distance import cosine

MichaelBrown = pandas.read_csv('IfTheyGunnedMeDown_frequencies.csv')
Ebola = pandas.read_csv('Ebola_frequencies.csv')
Cities = pandas.read_csv('USTop10Cities_frequencies.csv')

def set_count(X):
    X['total_count'] = X.apply(lambda r: r['w1'] + r['w2'] + r['w3'] + r['w4'], axis=1)

for ele in (Ebola, MichaelBrown, Cities):
    ele.columns = ['term', 'w1', 'w2', 'w3', 'w4']
    set_count(ele)

EbolaMichaelBrown = pandas.merge(Ebola, MichaelBrown, how='inner', on=['term'])
print EbolaMichaelBrown.sort('total_count_x', ascending=False)[:20][['term', 'total_count_x', 'total_count_y']]

MichaelBrownCities = pandas.merge(MichaelBrown, Cities, how='inner', on=['term'])
print MichaelBrownCities.sort('total_count_x', ascending=False)[:20][['term', 'total_count_x', 'total_count_y']]

EbolaCities = pandas.merge(Ebola, Cities, how='inner', on=['term'])
print EbolaCities.sort('total_count_x', ascending=False)[:20][['term', 'total_count_x', 'total_count_y']]

for ele in (EbolaMichaelBrown, EbolaCities, MichaelBrownCities):
    ele['total'] = ele.apply(lambda r: r['total_count_x'] + r['total_count_y'], axis=1)

print EbolaMichaelBrown.sort('total', ascending=False)[:20][['term', 'total']]
print EbolaCities.sort('total', ascending=False)[:20][['term', 'total']]
print MichaelBrownCities.sort('total', ascending=False)[:20][['term', 'total']]

EbolaMichaelBrownCities = pandas.merge(pandas.merge(Ebola, MichaelBrown, how='outer', on=['term']), Cities, how='outer', on=['term'])

# If there's any missing values, set them all to 0.
for val in ('_x', '_y', ''):
    for ele in (1,2,3,4):
        key = 'w%d%s' % (ele, val)
        EbolaMichaelBrownCities.loc[EbolaMichaelBrownCities[key].isnull(), key] = 0

# calculate cosine similarity between all 12 vectors
A = numpy.zeros((12,12))
i = 0
for suffix in ('_x', '_y', ''):
    for k in (1,2,3,4):
        key1 = 'w%d%s' % (k, suffix)
        j = 0
        for suffix2 in ('_x', '_y', ''):
            for k2 in (1,2,3,4):
                if i == j:
                    A[i, j] = 1
                elif i > j:
                    A[i, j] = A[j, i] # symmetric
                else:
                    key2 = 'w%d%s' % (k2, suffix2)
                    A[i, j] = cosine(EbolaMichaelBrownCities[key1], EbolaMichaelBrownCities[key2])

                j += 1
        i += 1
datasets = ("Ebola", "Ferguson", "Cities")
weeks = ("w1", "w2", "w3", "w4")
cols = []
for d in datasets:
    for w in weeks:
        cols.append("%s %s" % (d, w))
df = pandas.DataFrame(A, columns=cols)
df.index = cols
df.to_csv('cosine_sims.csv')



idx = []
for val in ('_x', '_y', ''):
    for ele in (1,2,3,4):
        key = 'w%d%s' % (ele, val)
        idx.append(key)
X = EbolaMichaelBrownCities[idx]

from sklearn.decomposition import PCA
import matplotlib.pyplot

# show cumulative sum of variances of each dimension of pca
# expect 3 dimensions for the 3 datasets but there seem to
# be more, i.e. weeks within each dataset aren't well correlated
pca = PCA()
Z = pca.fit_transform(X.T)
matplotlib.pyplot.plot(numpy.cumsum(pca.explained_variance_ratio_))
matplotlib.pyplot.title('Cumulative Varience Sum of Each PCA Dimension')
matplotlib.pyplot.xlabel('Weeks')
matplotlib.pyplot.ylabel('Sum')
matplotlib.pyplot.show()

