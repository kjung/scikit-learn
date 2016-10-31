################################################################################
# Run through 500 iterations of simulation 2 for double sample forests.
################################################################################

# CHANGE ME TO WORKING DIRECTORY WITH test_pf.py
setwd('~/Dropbox/personalized-predictions')

# How many iterations?
K <- 500
mse <- numeric(K)
coverage <- numeric(K)

# Parameters for the simulation
eta <- function(x) {
  return (1 + (1 / (1 + exp((-20*x - 1)/3)))) 
}  
d <- 8
n <- 5000

for (k in 1:K) {
  cat(sprintf("Generating data for %d\n", k))
  d <- 30
  n <- 500 
  X <- matrix(runif(d*n), nrow=n, ncol=d)
  propensities <- 0.25 * (1 + pbeta(X[,1], 2, 4))
  probs <- runif(n)
  W <- numeric(n)
  W[probs <= propensities] <- 1
  Y <- X[,1] - 0.5 + rnorm(n)
  train.data <- cbind(numeric(n), Y, W, X)
  colnames(train.data) <- c('tau', 'Y', 'W', paste('X', c(1:d)))
  write.table(train.data,
              file="data/sim1_train.txt",
              sep="\t",
              row.name=F,
              col.name=T)

  # Now generate a test dataset
  X <- matrix(runif(d*n), nrow=n, ncol=d)
  propensities <- 0.25 * (1 + pbeta(X[,1], 2, 4))
  probs <- runif(n)
  W <- numeric(n)
  W[probs <= propensities] <- 1
  Y <- X[,1] - 0.5 + rnorm(n)
  test.data <- cbind(numeric(n), Y, W, X)
  colnames(test.data) <- c('tau', 'Y', 'W', paste('X', c(1:d)))
  write.table(test.data,
              file="data/sim1_test.txt",
              sep="\t",
              row.name=F,
              col.name=T)

  # Run test_pf.py
  cat(sprintf("Fitting forest\n"))
  cmd <- "python test_pf.py"
  system(cmd)

  cat(sprintf("Reading in data.\n"))
  
  # Now read in the results, etc.  
  test.data <- read.table('data/sim2_test.txt', sep="\t", header=T)
  tau <- test.data$tau
  Y <- test.data$Y
  W <- test.data$W
  estimates <- read.table("propensity_forest_estimates_sim2.txt", header=F)[,1]
  mse[k] <- mean((tau - estimates)^2)
  # 0.0058 - so very good...  A definite bias, but not clear what that is about.

  # Okay, so the next step is to look at confidence intervals; this should be easy enough.
  # We'll just calculate the confidence intervals and estimate the coverage from that.  

  variances <- read.table('propensity_forest_variances_sim2.txt', sep="\t", header=F)[,1]
  sds <- sqrt(variances)
  sems <- sds / sqrt(2000)
  lower.ci <- estimates - 1.96*sems
  upper.ci <- estimates + 1.96*sems

  # Okay, tau = 0, so we just want to know how often the confidence intervals span 0...
  coverage[k] <- sum(lower.ci < tau & tau < upper.ci)/length(tau)
  
}  

save.image('run-pf-test-image.rda')

q(save='no')

