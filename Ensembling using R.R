library(mlbench)
library(caret)
library(caretEnsemble)

#====================================================================#
# Read in the data
#====================================================================#
data(Ionosphere)
dataset = Ionosphere
View(dataset)

#====================================================================#
# Format the data
#====================================================================#

str(dataset)

# Convert first column to numeric from factor
dataset$V1 = as.numeric(dataset$V1)

# Drop the 2nd column as it's all 0's
dataset[,2] = NULL


#====================================================================#
# Build the BOOSTING models
#====================================================================#

# We will try two models
#   1) C5.0 algorithm
#   2) Stochastic Gradient Boosting

control = trainControl(method = 'repeatedcv',
                       number = 10,
                       repeats = 5,
                       verbose=TRUE)
metric = 'Accuracy'
seed = 7

# Build a C5.0 model
#-------------------
set.seed(seed)
mod_C5.0 = train(Class ~., data=dataset, method='C5.0', metric=metric, trControl=control)


# Build a Stochastic Gradient Boosting model
#-------------------------------------------
set.seed(seed)
mod_GBM = train(Class ~., data=dataset, method='gbm', metric=metric, trControl=control)


# Summarise the results from both the models
#-------------------------------------------
boosting_results = resamples(list(C5.0=mod_C5.0, GBM=mod_GBM))
summary(boosting_results)
dotplot(boosting_results)




#====================================================================#
# Build the BAGGING models
#====================================================================#

# We will try two models
#   1) Bagged CART
#   2) Random Forest

control = trainControl(method = 'repeatedcv',
                       number = 10,
                       repeats = 3,
                       verbose=TRUE)
metric = 'Accuracy'
seed = 7

# Build a Bagged CART model
#--------------------------
set.seed(seed)
mod_Treebag = train(Class ~., data=dataset, method='treebag', metric=metric, trControl=control)


# Build a Random Forest model
#----------------------------
set.seed(seed)
mod_RF = train(Class ~., data=dataset, method='rf', metric=metric, trControl=control)


# Compare the model summaries
#----------------------------

bagging_results = resamples(list(Treebag = mod_Treebag, Random_Forest = mod_RF))
summary(bagging_results)
dotplot(bagging_results)



#====================================================================#
# Build a stacking algorith to combine multiple models
#====================================================================#

# We will stack the following models
#   1) CART
#   2) Logistic Regression
#   3) K-Nearest Neighbours
#   4) Support Vector Machines with a Radial Basic Kernel

# Set the control parameters
control = trainControl(method='repeatedcv',
                       number=10, repeats=3,
                       savePredictions = TRUE,
                       classProbs = TRUE,
                       verbose=TRUE)

algoList = c('rpart','glm','knn','svmRadial')


# Create the mmodels
set.seed(7)
models = caretList(Class ~ ., 
                   data=dataset, 
                   trControl = control, 
                   methodList = algoList)

# check the results for the models
results = resamples(models)
summary(results)
dotplot(results)

# Check the correlation between the models (ideally the models should have low correlations)
modelCor(results)
splom(results)


# Stack the models using Random Forest
stackControl = trainControl(method='repeatedcv', 
                            number=10, repeats=3,
                            savePredictions = TRUE,
                            classProbs = TRUE,
                            verbose=TRUE)

set.seed(7)

mod_Stack = caretStack(models, method='rf', metric='Accuracy', trControl = stackControl)
print(mod_Stack)


# We get an impressive accuracy of 93.9% with the stacked model using Random Forest!!!

# Hope you enjoyed this article