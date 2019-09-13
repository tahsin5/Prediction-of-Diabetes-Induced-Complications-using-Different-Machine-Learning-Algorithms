df <- read.csv(file="../Dataset/dataset_v1.csv", header=TRUE, sep=",")


##missForest
# install.packages("missForest")
library(missForest)

Sex<- factor(df$Sex, levels = c("1","2"))
df$Drinking.habit<- factor(df$Drinking.habit, levels = c("0","1","2"))
df$Smoking.habit<- factor(df$Smoking.habit, levels = c("0","1","2"))

df$Urinary.protein.at.0M<- factor(df$Urinary.protein.at.0M, levels = c("0","1","2","3","4"))
df$Urinary.protein.at.1M<- factor(df$Urinary.protein.at.1M, levels = c("0","1","2","3","4"))
df$Urinary.protein.at.3M<- factor(df$Urinary.protein.at.3M, levels = c("0","1","2","3","4"))
df$Urinary.protein.at.12M<- factor(df$Urinary.protein.at.12M, levels = c("0","1","2","3","4"))

df$Urinary.sugar.at.0M<- factor(df$Urinary.sugar.at.0M, levels = c("0","1","2","3","4","5"))
df$Urinary.sugar.at.1M<- factor(df$Urinary.sugar.at.1M, levels = c("0","1","2","3","4","5"))
df$Urinary.sugar.at.3M<- factor(df$Urinary.sugar.at.3M, levels = c("0","1","2","3","4","5"))
df$Urinary.sugar.at.12M<- factor(df$Urinary.sugar.at.12M, levels = c("0","1","2","3","4","5"))

df$History.of.complications<- factor(df$History.of.complications, levels = c("0","1"))
df$History.of.hypertension<- factor(df$History.of.hypertension, levels = c("0","1"))
df$History.of.dyslipidemia<- factor(df$History.of.dyslipidemia, levels = c("0","1"))
df$History.of.hyperuricemia<- factor(df$History.of.hyperuricemia, levels = c("0","1"))
df$History.of.retinopathy<- factor(df$History.of.retinopathy, levels = c("0","1"))
df$History.of.arteriosclerosis.obliterans<- factor(df$History.of.arteriosclerosis.obliterans, levels = c("0","1"))
df$History.of.atrial.fibrillation<- factor(df$History.of.atrial.fibrillation, levels = c("0","1"))
df$History.of.kidney.disease<- factor(df$History.of.kidney.disease, levels = c("0","1"))
df$History.of.liver.disease<- factor(df$History.of.liver.disease, levels = c("0","1"))
df$History.of.myocardial.infarction<- factor(df$History.of.myocardial.infarction, levels = c("0","1"))
df$History.of.angina.pectoris<- factor(df$History.of.angina.pectoris, levels = c("0","1"))
df$History.of.heart.failure<- factor(df$History.of.heart.failure, levels = c("0","1"))
df$History.of.cerebral.infarction<- factor(df$History.of.cerebral.infarction, levels = c("0","1","2"))

missforest_imputed <- missForest(df)
missforest_values <- missforest_imputed[["ximp"]]

write.csv(missforest_values, file = "D:/Documents/BRACU/Thesis/Dataset/missForest_imputed.csv")

error <- missforest_imputed[["OOBerror"]]
write.table(error, file = "D:/Documents/BRACU/Thesis/Code/missForest_imputed_100_error.txt", append = FALSE, sep = " ", dec = ".")


