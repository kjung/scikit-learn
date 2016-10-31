import numpy as np
import sklearn as sk
from sklearn.ensemble.forest import PropensityForest

################################################################################
# Sim2 from Wager & Athey
################################################################################
train_file = "/Users/kjung/Dropbox/personalized-predictions/data/sim1_train.txt"
train_data = np.loadtxt(fname=train_file,skiprows=1)
test_file = "/Users/kjung/Dropbox/personalized-predictions/data/sim1_test.txt"
test_data = np.loadtxt(fname=test_file,skiprows=1)

tau = train_data[:,0]
y = train_data[:,1]
w = train_data[:,2]
X = train_data[:,3:]

B = 1000

print "Fitting propensity forest with %d base trees..." % (B)
pf = PropensityForest(random_state=0,
                      n_estimators=B,
                      min_samples_leaf=1)
pf.fit(X=X, y=y, w=w, subsample_size=50,)
print "Getting estimates..."

estimates = pf.predict_effect(test_data[:,3:])
fname = "propensity_estimates_sim1.txt"
np.savetxt(fname=fname, X=estimates)

# Variance estimates for calculating confidence intervals.
print "Estimating variance..."
variances = pf.estimate_variance(test_data[:,3:])
fname = "propensity_variances_sim1.txt"
np.savetxt(fname, X=variances)
