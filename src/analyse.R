setwd("D:/YandexDisk/Analyses/borrowability-on-features/src")

library(glmnet)
library(randomForest)
library(pROC)

data <- read.csv('borrowing_w_features_detailed.csv', row.names = 1)
X <- model.matrix(outcome ~ .^2, data = data)
y <- data[,60]

# Takes too long
cv.fit <- cv.glmnet(X, y, family='binomial')
single.fit <- glmnet(X, y, family = 'binomial', lambda = cv.fit$lambda.1se)

data$outcome <- as.factor(data$outcome)
rf.fit <- randomForest(outcome ~ ., data = data, importance = T)

data.small <- read.csv('borrowing_w_features_small.csv', row.names = 1)
data.small$outcome <- factor(data.small$outcome)

# Leave only target-segment features
data.smaller <- data.small[,-7]
data.smaller <- data.smaller[,-6]
data.smaller <- data.smaller[,-4]
data.smaller$outcome <- factor(data.smaller$outcome)

# Predictive-importance test
rf.fit.smaller <- randomForest(outcome ~ .,
                               data = data.smaller,
                               importance = T)
varImpPlot(rf.fit.smaller, main = 'Target-segment features only',
           n.var = 10)

# Logistic-regression
# We remove intercept, which is not needed for glmnet
X.smaller <- model.matrix(outcome ~ ., data = data.smaller)[,-1]
y.smaller <- data.smaller$outcome
cv.fit.smaller <- cv.glmnet(X.smaller, y.smaller, family = 'binomial')
single.fit.smaller <- glmnet(X.smaller, y.smaller,
                             family = 'binomial', 
                             lambda = cv.fit.smaller$lambda.min)

old.par = par()


coef.names <- rownames(coef(single.fit.smaller))
coefs <- as.vector(coef(single.fit.smaller))
names(coefs) <- coef.names
coefs <- sort(coefs)

par(old.par)

pdf('simple_model.pdf', width = 8, height = 5)
par(mar = c(5.1, 12.1, 4.1, 2.1))
barplot(height = coefs[-1], 
        # names = predictor.names[-1],
        horiz = T, las = 2, cex.names=.5,
        cex.axis = .7)
dev.off()

rf.fit.smaller <- randomForest(x = X.smaller, y = y.smaller, importance = T)

pdf('simple_model_rf.pdf', width = 12)
varImpPlot(rf.fit.smaller, main = '')
dev.off()

X.small <- model.matrix(outcome ~ . + same_place*same_voice*same_manner,
                        data = data.small)[,-1]
y.small <- data$outcome
cv.fit.small <- cv.glmnet(X.small, y.small, family='binomial')
single.fit.small <- glmnet(X.small, y.small, family = 'binomial',
                           lambda = cv.fit.small$lambda.min)

coef.names <- rownames(coef(single.fit.small))
coefs <- as.vector(coef(single.fit.small))
names(coefs) <- coef.names
coefs <- sort(coefs)
pdf('bigger_model.pdf', width = 12, height = 7)
par(mar = c(5.1, 11.1, 4.1, 2.1))
barplot(height = coefs[-1], 
        # names = predictor.names[-1],
        horiz = T, las = 2, cex.names=.7,
        cex.axis = .7)
dev.off()

rf.fit.small <- randomForest(outcome ~ ., data = data.small, importance = T)

pdf('bigger_model_rf.pdf', width = 12, heigh = 8)
varImpPlot(rf.fit.small, main = '')
dev.off()

# Compare the models using cross-validation
set.seed(42)
n.fold <- 10
indices <- sample(1:nrow(data.small))
fold.size <- floor(nrow(data.small) / n.fold)
rf.aucs <- rep(0, n.fold)
glmnet.aucs <- rep(0, n.fold)
for (fold in 1:n.fold) {
  start.factor <- fold - 1
  cat(sprintf("Fold %d\n", 
              fold))
  if (fold == 1) {
    test.indices  <- indices[1:fold.size]
    train.indices <- indices[(fold.size+1):length(indices)]
    cat(sprintf("# train indices: %d; # test indices: %d\n", 
                length(train.indices),
                length(test.indices)))
  } else if (fold == n.fold) {
    start.i       <- fold.size * start.factor + 1
    test.indices  <- indices[start.i:length(indices)]
    train.indices <- indices[1:(start.i)-1]
    cat(sprintf("# train indices: %d; # test indices: %d\n", 
                length(train.indices),
                length(test.indices)))
  } else {
    start.i       <- fold.size * start.factor + 1
    end.i         <- start.i + fold.size
    test.indices  <- indices[start.i:end.i]
    train.indices <- c(
      indices[1:(start.i-1)],
      indices[(end.i+1):length(indices)])
    cat(sprintf("# train indices: %d; # test indices: %d\n", 
                length(train.indices),
                length(test.indices)))
  }
  stopifnot(length(test.indices) + length(train.indices) == length(indices))
  
  test.data <- data.small[test.indices,]
  test.X    <- model.matrix(outcome ~ . + same_place*same_voice*same_manner,
                            data = test.data)[,-1]
  test.y    <- test.data$outcome
  # print(test.y[1:20]); print(length(test.y))

  
  train.data   <- data.small[train.indices,]
  train.X      <- model.matrix(outcome ~ . + same_place*same_voice*same_manner,
                               data = train.data)[,-1]
  train.y      <- train.data$outcome
  
  train.rf.fit <- randomForest(outcome ~ ., data = train.data)
  train.cv.fit <- cv.glmnet(x = train.X, y = train.y, family = 'binomial')
  
  test.predictions.rf     <- predict(train.rf.fit, 
                                     test.data, type = 'prob')[,2]
  test.predictions.glmnet <- predict(train.cv.fit, 
                                     test.X, 
                                     s = 'lambda.min',
                                     type = 'response')
  rf.auc <- auc(roc(test.y, as.vector(test.predictions.rf)))
  glmnet.auc <- auc(roc(test.y, as.vector(test.predictions.glmnet)))
  # print(test.predictions[1:20]); print(length(test.predictions))
  # rf.accuracy <- sum(test.predictions.rf == test.y) / length(test.indices)
  # glmnet.accuracy <- sum(
  #   (test.predictions.glmnet > 0.5) == test.y) / length(test.indices)
  cat(sprintf("Glmnet AUC: %f; Random forest AUC: %f\n", 
              glmnet.auc,
              rf.auc))
  rf.aucs[fold] <- rf.auc
  glmnet.aucs[fold] <- glmnet.auc
}
cat(sprintf("Mean glmnet AUC: %f\n", mean(glmnet.aucs)))
cat(sprintf("Mean random forest AUC: %f\n", mean(rf.aucs)))


# data.binarised <- data.small
# data.binarised$same_manner <- as.numeric(data.binarised$same_manner > 0)
# data.binarised$same_place  <- as.numeric(data.binarised$same_place > 0)
# data.binarised$same_voice  <- as.numeric(data.binarised$same_voice > 0)
# rf.fit.small.binarised <- randomForest(outcome ~ ., 
#                                        data = data.binarised, 
#                                        importance = T)
