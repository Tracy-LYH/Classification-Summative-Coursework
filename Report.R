install.packages("DataExplorer")
install.packages("skimr")
install.packages("mlr3verse")
install.packages("psych")
install.packages("ranger")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("rpart.plot")
install.packages("data.table")
install.packages("randomForestSRC")
install.packages("xgboost")
install.packages("remotes")
remotes::install_github("mlr-org/mlr3extralearners")

library(skimr)
library(dplyr)
library(ggplot2)
library(psych)
library(data.table)
library(mlr3verse)
library(randomForestSRC)
library(ranger)
library(mlr3extralearners)
library(xgboost)
library(rpart.plot)

#data
bank <- read.table(file="https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv", header=TRUE, sep=",")
head(bank)
dim(bank)

# the summary of data
skim(bank)

#check missing data
missing_values_count <- colSums(is.na(bank))
missing_values_count#no missing data

# delete invalid data
exp_invalid <- sum(bank$Experience < 0)
code_invalid <- sum(bank$ZIP.Code < 90000)
bank_data <- bank[bank$Experience >= 0, ]
bank_data <- bank_data[bank_data$ZIP.Code >= 90000,]

# the summary of new data
skim(bank_data)

# EDA
#pie
loan_summary <- bank_data %>%
  count(Personal.Loan) %>%
  pull(n)
lable_loan <- c( "not accept","accept")
piepercent<- round(100*loan_summary/sum(loan_summary), 1)
pie(loan_summary, labels = piepercent, main = "Loan Pie Chart",col = c("blue","gray"))
legend("bottomright", c("not accept", "accept"), cex = 0.8, fill = c("blue","gray"))

# boxplot
numeric_vars <- c("Age","Education","CCAvg","Experience","Family","Income","Mortgage","ZIP.Code")
par(mfrow=c(2, 4))
for(var_name in numeric_vars) {
  boxplot_formula <- as.formula(paste(var_name, "~ bank_data$Personal.Loan"))
  boxplot(boxplot_formula,
          data = bank_data,
          main = paste(var_name),
          xlab = "Personal Loan",
          ylab = var_name,
          names = c("Not Accept", "Accept"))
}

# correlation
corPlot(bank_data)

#model fitting
df <- as.data.frame(bank_data)
df$Securities.Account = as.factor(df$Securities.Account)
df$Education = as.factor(df$Education)
df$CD.Account = as.factor(df$CD.Account)
df$Online = as.factor(df$Online)
df$CreditCard = as.factor(df$CreditCard)
df$Personal.Loan = as.factor(df$Personal.Loan)

set.seed(10) 
task <- TaskClassif$new(id="back",backend = df, target = "Personal.Loan")

#build models
learner_baseline <- lrn("classif.featureless", predict_type = "prob")
learner_cart <- lrn("classif.rpart", predict_type = "prob")
learner_rf <- lrn("classif.rfsrc", predict_type = "prob")
learner_log_reg <- lrn("classif.log_reg", predict_type = "prob")
learner_naive_bayes <- lrn("classif.naive_bayes", predict_type = "prob")
learner_cart_cp <- lrn("classif.rpart", predict_type = "prob")
learner_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%po(learner_xgboost)


#cross-validation
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task)

#cart
learner_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(task, learner_cart_cv, cv5, store_models = TRUE)
for (i in 1:5) {
  rpart::plotcp(res_cart_cv$learners[[i]]$model)
}


# model set
res <- benchmark(data.table(
  task       = list(task),
  learner    = list(learner_baseline,
                    learner_cart,
                    learner_rf,
                    learner_cart_cp,
                    learner_log_reg,
                    learner_naive_bayes,
                    pl_xgb),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

#random forest
learner_rf <- lrn("classif.rfsrc", predict_type = "prob", importance = 'TRUE')
res_rf <- resample(task, learner_rf, cv5, store_models = TRUE)
rf_model <- res_rf$learners[[3]]$model
#rf_model
#plot(rf_model)

# #plot importance
# important_scores <- res_rf$learners[[1]]$importance()
# for (i in 2:5) {important_scores <- important_scores + res_rf$learners[[i]]$importance()}
# 
# important_scores <- important_scores/5
# lables <- c("Income","Education","Family", "CCAvg", "CD.Account","Mortgage", "Age", "Experience", "ZIP.Code", "CreditCard", "Securities.Account", "Online")
# scores_sort <- sort(important_scores,decreasing = TRUE)
# piepercent<- round(100*scores_sort, 1)
# pie(scores_sort[1:7], labels =piepercent, main = "Importance", col = rainbow(length(lables)))
# legend("topright", c("Income", "Education", "Family", "CCAvg", "CD.Account", "Mortgage"), cex = 0.8, fill = rainbow(length(lables)))


#tree plot
trees <- res$resample_result(2)
tree1 <- trees$learners[[3]]
tree1_rpart <- tree1$model
rpart.plot(tree1_rpart)

#tune
tune_ps_ranger <- ps(mtry = p_int(lower = 1, upper = 6))
evals_trm = trm("evals", n_evals = 25)

instance_ranger <- TuningInstanceSingleCrit$new(task = task,learner = learner_rf,resampling = cv5,measure = msr("classif.ce"),search_space = tune_ps_ranger,terminator = evals_trm)

tuner <- tnr("grid_search", resolution = 5)
tuner$optimize(instance_ranger) 
instance_ranger$result_learner_param_vals

#refit
set.seed(10)
final_model <- lrn("classif.rfsrc", predict_type = "prob", importance = 'TRUE')
final_model$param_set$values = list(mtry = 3, ntree = 24)
res_rf <- resample(task, learner_rf, cv5, store_models = TRUE)

res_rf$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.auc"),
                      msr("classif.fpr"),
                      msr("classif.fnr")))
