packageSet <- c("car", "abind", "aplpack", "colorspace", "effects", "Hmisc",
                "leaps", "zoo", "lmtest", "mvtnorm", "multcomp", "relimp", "rgl", "RODBC",
                "clv", "rpart.plot", "flexclust", "e1071", "sem", "Rcmdr", "BCA","foreign","AMORE","tree","rpart","rattle","ipred","randomForest","dplyr","sqldf","genalg")
install.packages(packageSet)
rm(packageSet)


library(BCA)
library(relimp)
library(car)
library(RcmdrMisc)
library(nnet)
library(foreign)

library(rpart)
library(rattle)
library(AMORE)
setwd("C:/Users/Ngoc/Desktop/Master 2/Big Data/project")
train <- read.csv("train_loan.csv", sep = ",")


pairs(train, panel = panel.smooth, main = "train")

train <- na.omit(train)
train <- within(train, {
  Loan_ID <- NULL 
})


summary(train)

variable.summary(train)


pairs(train, panel = panel.smooth, main = "train")

# Creating validation and estimation subsets

train$Sample <- create.samples(train, est = 0.7, val = 0.3)


# Creating numerical variable from factor variable MonthsGive

train <- within(train, {
  Loan_Status.Num <- Recode(Loan_Status, '"Y"=1; "N"=0', as.factor.result=FALSE)
})

train <- within(train, {
  Loan_Status.NF <- Recode(Loan_Status, '"Y"=1; "N"=0', as.factor.result=TRUE)
})

GLM.2 <- glm(Loan_Status ~ ApplicantIncome + CoapplicantIncome + LoanAmount 
            +Loan_Amount_Term + Credit_History, family=binomial(logit), data=train)
summary(GLM.2)
# R2 McFadden value
1 - (GLM.2$deviance/GLM.2$null.deviance) # McFadden R2

# Building non linear model

numSummary(train[,c("ApplicantIncome", "CoapplicantIncome","LoanAmount")])

# Logarithm of numerical variables

# In case of variable CoapplicantIncome we are using log(X+1)
train$LogCoapplicantIncome <- with(train, log(CoapplicantIncome+1))

train$LogApplicantIncome <- with(train, log(ApplicantIncome))
train$LogLoanAmount <- with(train, log(LoanAmount))

# Building nnlinear model


LogCCS <- glm(Loan_Status ~ Gender + Married + Dependents + Education + Self_Employed + LogApplicantIncome + LogCoapplicantIncome + LogLoanAmount 
              +Loan_Amount_Term + Credit_History + Property_Area, family=binomial(logit), 
              data=train, subset=Sample=="Estimation")
summary(LogCCS)
1 - (LogCCS$deviance/LogCCS$null.deviance) # McFadden R2

# Checking p-values for factor variables

Anova(LogCCS)

#  Building mixed model


MixedCCS <- glm(Loan_Status ~ LogApplicantIncome + CoapplicantIncome + LogLoanAmount 
                +Loan_Amount_Term + Credit_History + Property_Area, family=binomial(logit), 
                data=train, subset=Sample=="Estimation")
summary(MixedCCS)
1 - (MixedCCS$deviance/MixedCCS$null.deviance) # McFadden R2

# Classification trees

library(rpart)
library(rattle)


train.rpart <- rpart(Loan_Status ~ ApplicantIncome + CoapplicantIncome + LoanAmount 
                     +Loan_Amount_Term + Property_Area,
                     data=train, cp=0.01, subset=Sample=="Estimation")

plot(train.rpart)
text(train.rpart)
print(train.rpart)
printcp(train.rpart)
plotcp(train.rpart)
fancyRpartPlot(train.rpart)

train.rpart2 <- rpart(Loan_Status ~ ApplicantIncome + CoapplicantIncome + LoanAmount 
                     +Loan_Amount_Term + Credit_History,
                     data=train, cp=0.008, subset=Sample=="Estimation")


plotcp(train.rpart2)
printcp(train.rpart2) # Pruning Table
train.rpart2 # Tree Leaf Summary
plot(train.rpart2, extra = 4, uniform = TRUE, fallen.leaves = FALSE)
text(train.rpart2)
fancyRpartPlot(train.rpart2)



mod <- randomForest(train$Loan_Status~., data=train,  ntrees=100)
a<-predict(mod, newdata=train)
a<-round(a,0)
result<- data.frame(a,train$Loan_Status)


b<-sum(abs(a-train$Loan_Status))




# Neural networks

install.packages("RcmdrPlugin.BCA")
install.packages("Rcmdr")
install.packages("splines")
install.packages("effects")
install.packages("grid")
install.packages("lattice")
install.packages("modeltools")
install.packages("stats4")

library("RcmdrPlugin.BCA")
library("Rcmdr")
library("splines")
library("effects")
library("grid")
library("lattice")
library("modeltools")
library("stats4")



NNET.2 <- Nnet(Loan_Status ~ Gender + Married + Dependents + Education + Self_Employed + LogApplicantIncome + LogCoapplicantIncome + LogLoanAmount 
               +Loan_Amount_Term + Credit_History + Property_Area, data=train, decay=0.10, 
               size=4, subset='Sample=="Estimation"')
NNET.2$value # Final Objective Function Value
summary(NNET.2)

lift.chart(c("GLM.2","MixedCCS","train.rpart2","NNET.2","mod"),train[train$Sample=="Estimation",],"Y", 0.01, 
           "cumulative", "Zbiór estymacyjny")


lift.chart(c("GLM.2","MixedCCS","train.rpart2","NNET.2","mod"),train[train$Sample=="Validation",],"Y", 0.01, 
           "cumulative", "Zbiór estymacyjny")

wynik<-predict(NNET.2,data=train, subset='Sample=="Validation"')
wynik<-round(wynik,0)

