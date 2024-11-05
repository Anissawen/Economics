# library(ISLR2)
library(dplyr)
library(keras3)
library(MASS)

# Data Setup

# NOTE: fitting is done in python, so setting the seed here does not
# ensure identical results when using keras3.
set.seed(22)
n <- 1000
beta <- seq(1:20)
beta <- matrix(beta, ncol=1)

X <- mvrnorm(n, mu = rep(0, 20), Sigma = diag(20))
epsilon <- rnorm(100, 0, 3)
y <- X%*%beta + epsilon
ntest <- trunc(n / 5)
testid <- sample(1:n, ntest)
X_train <- X[-testid,]
y_train <- y[-testid]
X_test <- X[testid, ]
y_test <- y[testid]

fit <- lm(y_train ~ 0 + X_train)
summary(fit)
beta_hat <- matrix(fit$coefficients, ncol=1)
l_pred <- X_test %*% beta_hat
mean(abs(y_test - l_pred))


nval <- trunc((n-ntest) / 4)
val_id <- sample(1:nrow(X_train), nval)
X_train_nn <- X_train[-val_id,]
y_train_nn <- y_train[-val_id]
X_val <- X_train[val_id,]
y_val <- y_train[val_id]


modnn <- keras_model_sequential(input_shape = ncol(X_train_nn)) %>%
  layer_dense(units = 1)

modnn %>% compile(loss = "mse",
                  optimizer = optimizer_rmsprop(learning_rate = 0.01),
                  metrics = metric_mean_squared_error())
                  #metrics = list("mean_absolute_error") )

history <- modnn %>% fit(
  X_train_nn, y_train_nn, epochs = 200, batch_size = 32,
  validation_data = list(X_val, y_val)
)

plot(history)

npred <- predict(modnn, X[testid, ])
mean(abs(y[testid] - npred))

mean(abs(l_pred - npred))
