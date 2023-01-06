

#-----read data-------------S
df <- read.csv("CHUUU.csv", header = TRUE, stringsAsFactors = TRUE)
head(df)

library(DataExplorer)
library(ggplot2)
library(tidyverse)

#--------initial data exploration---------
summary(df)
str(df)

#changing variable from numeric to factor
df$Area.code <- as.factor(df$Area.code)
plot_str(df)

#------data preprocessing------
df <- prodNA(df, noNA = 0.05)
sum(is.na(df))
plot_missing(df)

#--------distribution of missing values in each variable
NAcol <- which(colSums(is.na(df)) > 0)
NAcol
df_na <- sort(colSums(sapply(df[NAcol], is.na)), decreasing = TRUE)
df_na

#---------imputation of missing values------------
'''Categorical missng value imputation using mode'''
df <- mutate_all(df, na_if, '')
i1 <- !sapply(df, is.numeric)
Mode <- function(x) {
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x,ux)))]
}
df[i1] <- lapply(df[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))
sum(is.na(df))


'''imputing contionous missing values'''
library(caret)
preProcvalues <- preProcess(df, method = "medianImpute")
df <- predict(preProcvalues, df)
sum(is.na(df))

'''checking for missing value'''
colSums(is.na(df))

#--------------data encoding and onehot encoding-------------------------------
str(df)
df$Area.code <- factor(df$Area.code, levels =c("408","415","510"), labels =c("1","2","3") )
df$International.plan <- factor(df$International.plan, levels =c("No","Yes"), labels =c("0","1") )
df$Voice.mail.plan <- factor(df$Voice.mail.plan, levels =c("No","Yes"), labels =c("0","1") )
df$Churn <- factor(df$Churn, levels =c("False","True"), labels =c("0","1") )
attach(df)
df

'''onehot encoding'''
df <- df[-c(1)]
dmy <- dummyVars(Churn ~ ., data = df)
dataset_onehot <- data.frame(predict(dmy, newdata = df))
data_new <- cbind(dataset_onehot,Churn)
str(data_new)

df <- data_new
str(df)
plot_correlation(df, type = c("all"))
head(df)

#-----droping variables-----
str(df)
df <- df[-c(9,10,13,16,19)]
plot_correlation(df,type =c("all"))

#----explanatory data analysis------
library(inspectdf)
df %>% 
  inspect_cat() %>% 
  show_plot()

df %>% 
  inspect_num() %>% 
  show_plot()

ggplot(filter(df, Churn == "False"), aes(x = Total.day.calls, fill = Churn)) +
  geom_line(position = "dodge", stat = "count", colour = "red") + theme_classic()
ggplot(filter(df, Churn == "True"), aes(x = Total.day.calls, fill = Churn)) +
  geom_line(position = "dodge", stat = count, colour = "red") +theme_classic()

plot_correlation(df, type = c("continuous"))


#-----balancing dataset------
table(df$Churn)
prop.table(table(df$Churn))
set.seed(100)

dim(df)
# upsampling data train
df$Churn <- as.numeric(df$Churn)
dfnew = SMOTE(df[-12], df$Churn)
dfnew = dfnew$data
dfnew$class <- as.factor(dfnew$class)
barplot(table(dfnew$class))
prop.table(table(dfnew$class))
str(dfnew)
data <- dfnew
str(data)
data <- data[-c(17)]

#--------------splitting the dataset------
set.seed(123)
split = sample.split(data$class, SplitRatio = 0.7)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)





#----------------------SVM----------------
svm_rbf <- svm(class~., data = train_set)
svm_linear = svm (class~., data = train_set, kernel = "linear")
svm_sigmoid = svm (class~., data = train_set, kernel = "sigmoid")
svm_polynomial = svm (class~., data = train_set, kernel = "poly")

summary(svm_rbf)
svm_rbf$gamma
pred <- predict(svm_rbf,train_set)
pred1 <- predict(svm_linear,train_set)
pred3 <- predict(svm_polynomial,train_set)


#-----confusion matrix training set----
confusionMatrix(table(pred,train_set$class))
confusionMatrix(table(pred1,train_set$class))
confusionMatrix(table(pred3,train_set$class))


#-----confusionmatrix test set-----
predd <- predict(svm_rbf,test_set)
predd1 <- predict(svm_linear,test_set)
predd3 <- predict(svm_polynomial,test_set)
confusionMatrix(table(predd,test_set$class))
confusionMatrix(table(predd1,test_set$class))
confusionMatrix(table(predd3,test_set$class))


#------roc----
test_set
train_set
pred <- as.numeric(pred)
pred1=prediction(pred,train_set$class) 

perf1=performance(pred1,"tpr","fpr") 

plot(perf1,colorize=T,main="train_set",ylab="Sensitivity",xlab="1-Specificity", 
     
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7)) 
#-----AOC-------
auc1=as.numeric(performance(pred1,"auc")@y.values) 
auc1=round(auc1,3) 
auc1 



#---------model improvement--------
set.seed(123)
# tune function tunes the hyperparameters of the model using grid search method
tuned_model = tune(svm, class~., data=train_set,
                   ranges = list(epsilon = seq (0, 1, 0.1), cost = 2^(0:3)))
plot (tuned_model)
summary (tuned_model)
tuned_model$best.parameters
opt_model = tuned_model$best.model
summary(opt_model)

# Building the best model
svm_best <- svm (class~., data = train_set, epsilon = 0, cost = 8)
summary(svm_best)
predd <- predict(svm_best,test_set)
confusionMatrix(table(predd,test_set$class))

#------roc----

predd <- as.numeric(predd)
pred1=prediction(predd,test_set$class) 

perf1=performance(pred1,"tpr","fpr") 

plot(perf1,colorize=T,main="train_set",ylab="Sensitivity",xlab="1-Specificity", 
     
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7)) 
#-----AOC-------
auc1=as.numeric(performance(pred1,"auc")@y.values) 
auc1=round(auc1,3) 
auc1


#---------decision trees------------
#Default split is with Gini index
tree = rpart(class~ ., data=train_set)
tree
rpart.plot(tree, extra = 101, nn = TRUE)

#evaluation of the model
model5 = predict(tree, train_set, type = "class")
table(model5, train_set$class)
confusionMatrix(table(model5, train_set$class))



model5 = predict(tree, test_set, type = "class")
table(model5, test_set$class)
confusionMatrix(table(model5, test_set$class))

model5 <- as.numeric(model5)
pred1=prediction(model5,train_set$class) 

perf1=performance(pred1,"tpr","fpr") 

plot(perf1,colorize=T,main="train_set",ylab="Sensitivity",xlab="1-Specificity", 
     
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7)) 
#-----AOC-------
auc1=as.numeric(performance(pred1,"auc")@y.values) 
auc1=round(auc1,3) 
auc1 

plotcp(tree_with_params)

#tuning the model
tuned = rpart(class ~ ., data=train_set, method="class", minsplit = 0.2, minbucket = 20, cp = 0.0029)

model5 = predict(tuned, train_set, type = "class")
table(model5, train_set$class)
confusionMatrix(table(model5, train_set$class))

model5 = predict(tuned, test_set, type = "class")
table(model5, test_set$class)
confusionMatrix(table(model5, test_set$class))

#-------ROC------
model5 <- as.numeric(model5)
pred1=prediction(model5,test_set$class) 

perf1=performance(pred1,"tpr","fpr") 

plot(perf1,colorize=T,main="ROC curve",ylab="Sensitivity",xlab="1-Specificity", 
     
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7)) 
#-----AOC-------
auc1=as.numeric(performance(pred1,"auc")@y.values) 
auc1=round(auc1,3) 
auc1 








#--------------Randomforest-------------------
set.seed(345)
Randomf <- randomForest(class~.,data = train_set )


summary(Randomf)

p1 <- predict(Randomf, test_set)
confusionMatrix(p1,test_set$class)

plot(Randomf)
P1 <- as.numeric(p1)
pred1=prediction(P1,test_set$class) 

perf1=performance(pred1,"tpr","fpr") 

plot(perf1,colorize=T,main="test_set",ylab="Sensitivity",xlab="1-Specificity", 
     
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7)) 
#-----AOC-------
auc1=as.numeric(performance(pred1,"auc")@y.values) 
auc1=round(auc1,3) 
auc1 






# Tuning mtry
tuneRF(train_set[ ,-17], train_set$class,
       stepFactor=0.5,
       plot = TRUE,
       ntreeTry = 800,
       trace = TRUE,
       improve = 0.05)



randomf1 <- randomForest(class~.,data = train_set,
                         ntreeTry = 800,
                         mtry=4,
                         importance = TRUE,
                         proximity = TRUE)
plot(randomf1)

p55 <- predict(randomf1, test_set)
confusionMatrix(p55,test_set$class)


p55 <- as.numeric(p55)
pred1=prediction(p55,test_set$class) 

perf1=performance(pred1,"tpr","fpr") 

plot(perf1,colorize=T,main="ROC curve",ylab="Sensitivity",xlab="1-Specificity", 
     
     print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7)) 
#-----AOC-------
auc1=as.numeric(performance(pred1,"auc")@y.values) 
auc1=round(auc1,3) 
auc1 





# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(100)
mtry <- sqrt(ncol(train()))
rf_random <- train(churn ~., data=train_set, method="rf", tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

p2 <- predict(rf_random, test_set)
confusionMatrix(p2,test_set$Churn)
























