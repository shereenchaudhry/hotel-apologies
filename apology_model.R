#read in the libraries and data
library('readr')
library('politeness')
library('dplyr')
library('DTMtools')
library('ggplot2')
data <- read_csv("existence.csv")

polite.data <- politeness(data$response, parser="spacy") #get politeness features with politeness package

#visualize politeness features if needed
#politeness::politenessPlot(polite.data,
#                           split=data$apology,
#                           #split_levels = c(0, 1),
#                           split_name = "Apology")


DV = data$apology #set DV
full_data = cbind(data, polite.data) #full data consists of LIWC, TextAnalyzer, and Politeness features

#remove unnecessary columns
full_data = full_data[,-c(1:23)]
full_data$num_ratings = NULL
full_data$mturk_rating = NULL
full_data$apology = NULL
full_data$Brand = NULL

#scale the data and remove columns with na after scaling (columns with only one value)
polite.data = data.frame(scale(polite.data))
polite.data = polite.data[ , colSums(is.na(polite.data)) == 0]
full_data = data.frame(scale(full_data))
full_data = full_data[ , colSums(is.na(full_data)) == 0]

#get Ngrams (1~3 grams) and scale
ngram.data<-data.frame(data$response %>% DTMtools::DTM(ngrams = 1:3, stop.words = TRUE))
ngram.data = data.frame(scale(ngram.data))

#initialize the 5 models (Ngram, Politeness, Full, Apology_Only, WordCount_only)
CV.models<-list(ngrams=list(exes=ngram.data), politeness=list(exes=polite.data), full=list(exes=full_data),
                apology=list(exes=full_data[['Apology']]), wordcount=list(exes=full_data[['WC']]), ngrams_w_dummies=list(NA),
                politeness_w_dummies=list(NA), full_w_dummies=list(NA), apology_w_dummies=list(NA), wordcount_w_dummies=list(NA))
#set number of cycles
cycles<-2

#initialize guesses and coefficients of models
for(x in 1:length(CV.models)){
CV.models[[x]][["guesses"]]<-array(NA,c(length(DV),cycles))
CV.models[[x]][["coefs"]]<-NA
}

#for the first three models, run CV Lasso and record the estimation made by the model, and its coefficients
for(x in 1:3){
for(cycle in 1:cycles){
cycleModel<-politenessProjection(df_polite_train = CV.models[[x]]$exes, covar=DV, cv_folds=10)
CV.models[[x]][["guesses"]][,cycle]<-cycleModel$train_proj
}
CV.models[[x]][["guess"]]<-rowMeans(CV.models[[x]][["guesses"]],na.rm=TRUE)
CV.models[[x]][["coefs"]]<-cycleModel$train_coefs
}

#for the models 4 and 5 (only one variable), run Logistic Regression
for(x in 4:5){
  for(cycle in 1:cycles){
    m_polite_train <- as.matrix(CV.models[[x]]$exes)
    foldIDs<-politeness:::foldset(length(DV), 10)
    tpb<-utils::txtProgressBar(0,10)
    polite_fit<-rep(NA,10)
    for(fold in 1:max(foldIDs)){ #N-fold CV
      train.fold<-(foldIDs!=fold) 
      test.fold<-(foldIDs==fold) 
      m_data = data.frame(y=DV[train.fold], x=m_polite_train[train.fold]) #get training data
      m_test = data.frame(x=m_polite_train[test.fold,]) #get test data
      polite_model_fold<-glm(y ~ x, data=m_data, family='binomial') #change family to 'gaussian' if non-binary
      polite_fit[test.fold]<-as.vector(predict(polite_model_fold, m_test, type="response")) #get prediction
      utils::setTxtProgressBar(tpb,fold)
    }     
    polite_model<-glm(DV ~ CV.models[[x]]$exes, family='binomial') #change family to 'gaussian' if non-binary
    p_coefs<-as.matrix(stats::coef(polite_model)) #get coefficients
    polite_coefs<-p_coefs[(!(rownames(p_coefs)=="(Intercept)"))&p_coefs!=0,]
    CV.models[[x]][["guesses"]][,cycle]<-polite_fit #get estimates in each cycle
  }
  CV.models[[x]][["guess"]]<-rowMeans(CV.models[[x]][["guesses"]],na.rm=TRUE) #record mean estimates
  CV.models[[x]][["coefs"]]<-polite_coefs #record coefficients
  CV.models[[x]][['summ']]<-summary(polite_model)
}


# Versions of Models with Dummy Variables
for(x in 6:8){
surviving = names(CV.models[[x-5]][['coefs']]) #get only columns chosen by CV Lasso
new_full_data = CV.models[[x-5]]$exes[surviving]
dummies = fastDummies::dummy_cols(data$Brand, remove_first_dummy = TRUE) #make dummies with Brand (or other variables)
dummies[[".data"]] = NULL
new_full_data = cbind(new_full_data, dummies) #set new matrix
#run Logistic Regression with N-fold CV
for(cycle in 1:cycles){
  m_polite_train <- as.matrix(new_full_data)
  foldIDs<-politeness:::foldset(length(DV), 10)
  tpb<-utils::txtProgressBar(0,10)
  polite_fit<-rep(NA,10)
  for(fold in 1:max(foldIDs)){
    train.fold<-(foldIDs!=fold)
    test.fold<-(foldIDs==fold)
    m_data = data.frame(y=DV[train.fold], x=m_polite_train[train.fold,])
    m_test = data.frame(x=m_polite_train[test.fold,])
    polite_model_fold<-glm(y ~ ., data=m_data, family='binomial') #change family to 'gaussian' if non-binary
    polite_fit[test.fold]<-as.vector(predict(polite_model_fold, m_test, type="response"))
    utils::setTxtProgressBar(tpb,fold)
  }     
  polite_model<-glm(DV ~ m_polite_train, family='binomial') #change family to 'gaussian' if non-binary
  p_coefs<-as.matrix(stats::coef(polite_model))
  polite_coefs<-p_coefs[(!(rownames(p_coefs)=="(Intercept)"))&p_coefs!=0,]
  CV.models[[x]][["guesses"]][,cycle]<-polite_fit
}
CV.models[[x]][["guess"]]<-rowMeans(CV.models[[x]][["guesses"]],na.rm=TRUE)
CV.models[[x]][["coefs"]]<-polite_coefs
CV.models[[x]][['summ']]<-summary(polite_model)
}

#identical to DV above
for(x in 9:10){
  dummies = fastDummies::dummy_cols(data$Brand, remove_first_dummy = TRUE)
  dummies[[".data"]] = NULL
  new_full_data = cbind(CV.models[[x-5]]$exes, dummies)
  for(cycle in 1:cycles){
    m_polite_train <- as.matrix(new_full_data)
    foldIDs<-politeness:::foldset(length(DV), 10)
    tpb<-utils::txtProgressBar(0,10)
    polite_fit<-rep(NA,10)
    for(fold in 1:max(foldIDs)){
      train.fold<-(foldIDs!=fold)
      test.fold<-(foldIDs==fold)
      m_data = data.frame(y=DV[train.fold], x=m_polite_train[train.fold,])
      m_test = data.frame(x=m_polite_train[test.fold,])
      polite_model_fold<-glm(y ~ ., data=m_data, family='binomial') #change family to 'gaussian' if non-binary
      polite_fit[test.fold]<-as.vector(predict(polite_model_fold, m_test, type="response"))
      utils::setTxtProgressBar(tpb,fold)
    }     
    polite_model<-glm(DV ~ m_polite_train, family='binomial') #change family to 'gaussian' if non-binary
    p_coefs<-as.matrix(stats::coef(polite_model))
    polite_coefs<-p_coefs[(!(rownames(p_coefs)=="(Intercept)"))&p_coefs!=0,]
    CV.models[[x]][["guesses"]][,cycle]<-polite_fit
  }
  CV.models[[x]][["guess"]]<-rowMeans(CV.models[[x]][["guesses"]],na.rm=TRUE)
  CV.models[[x]][["coefs"]]<-polite_coefs
  CV.models[[x]][['summ']]<-summary(polite_model)
}

#get binary estimates based on 'guess' column for each of the model and record accuracy
for(x in 1:10){
CV.models[[x]][['est']] <- CV.models[[x]][['guess']] > 0.5
CV.models[[x]][['accuracy']] = sum(data$apology == CV.models[[x]][['est']]) / dim(data)[1]
}
for(x in 1:10){print(CV.models[[x]][['accuracy']])}

#visualize each coefficients (first three models only)
i = 1
df = data.frame(name=rownames(data.frame(CV.models[[i]]['coefs'])), 
                coefs=round(data.frame(CV.models[[i]]['coefs'])['coefs'],2))
rownames(df) = NULL
ggplot(data=df, aes(x=name, y=coefs)) +
  geom_bar(stat="identity", fill="blue")+
  geom_text(aes(label=coefs), vjust=-0.5, color="black", size=3.5)+labs(x='Feature')+labs(y='Coef')+labs(title='Coef of Each Feature')+theme_bw()+
  theme(axis.text=element_text(size=10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title=element_text(size=13,face="bold"),
        plot.title=element_text(size=20, face='bold', hjust=0.5))

i = 2
df = data.frame(name=rownames(data.frame(CV.models[[i]]['coefs'])), 
                coefs=round(data.frame(CV.models[[i]]['coefs'])['coefs'],2))
rownames(df) = NULL
ggplot(data=df, aes(x=name, y=coefs)) +
  geom_bar(stat="identity", fill="blue")+
  geom_text(aes(label=coefs), vjust=-0.5, color="black", size=3.5)+labs(x='Feature')+labs(y='Coef')+labs(title='Coef of Each Feature')+theme_bw()+
  theme(axis.text=element_text(size=10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title=element_text(size=13,face="bold"),
        plot.title=element_text(size=20, face='bold', hjust=0.5))

i = 3
df = data.frame(name=rownames(data.frame(CV.models[[i]]['coefs'])), 
                coefs=round(data.frame(CV.models[[i]]['coefs'])['coefs'],2))
rownames(df) = NULL
ggplot(data=df, aes(x=name, y=coefs)) +
  geom_bar(stat="identity", fill="blue")+
  geom_text(aes(label=coefs), vjust=-0.5, color="black", size=3.5)+labs(x='Feature')+labs(y='Coef')+labs(title='Coef of Each Feature')+theme_bw()+
  theme(axis.text=element_text(size=10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title=element_text(size=13,face="bold"),
        plot.title=element_text(size=20, face='bold', hjust=0.5))
