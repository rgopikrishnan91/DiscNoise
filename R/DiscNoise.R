library(randomForest.ddR)
library(randomForest)
library(pROC)
library(ROCR)
library(doMC)
library(data.table)
library(rms)
library(caret)
library(mccr)
library(foreign)



#Extract importance from the models

get_importance_generic<-function(model,var){
  if(classifier=='rf'){

    return(importance(model)[,'MeanDecreaseAccuracy'])
  }else{
  m<-varImp(model)$importance
  m$class2<-NULL
  res <- m[match(var,rownames(m)),]
  return(res)
  }
  
}

### Generating OOSB samples 
create_oosb_sample<-function(data,boot_size,seed=TRUE){
  
  train_indices <- list()
  test_indices <- list()
  
  #For reproducability
if(is.numeric(seed)){
  set.seed(seed)
}else if(seed==TRUE){
  set.seed(42)
}else{
  seed<-FALSE
}
  
  for(count in 1:boot_size){
    data <- data[sample(nrow(data)),]
    train_index<-sample(seq(1:nrow(data)),size=nrow(data),replace=T)
    test_index<-seq(1:nrow(data))[-train_index]
    
    train_indices[[count]]<-train_index
    test_indices[[count]]<-test_index
  }
  
  return(list(train_indices,test_indices))
  
}

#Extract Auc
get_auc<-function(actuals,predicted){
  predictions<-prediction(predicted,actuals)
  
  auc<-ROCR::performance(predictions,'auc')
  auc<-unlist(slot(auc,'y.values'))
  result_auc<-min(round(auc,digits=2))
  return(result_auc)
}

#Function to get Mathew's Correlation Coefficient 
get_mcc<-function(act,pred,classes)
{
  TP <- sum(act == classes[1] & pred == classes[1])
  TN <- sum(act == classes[2] & pred == classes[2])
  FP <- sum(act == classes[2] & pred == classes[1])
  FN <- sum(act == classes[1] & pred == classes[2])
  denom <- as.double(TP + FP) * (TP + FN) * (TN + FP) * (TN + 
                                                           FN)
  if (any((TP + FP) == 0, (TP + FN) == 0, (TN + FP) == 0, (TN + 
                                                           FN) == 0)) 
    denom <- 1
  mcc <- ((TP * TN) - (FP * FN))/sqrt(denom)
  return(mcc)
}



#This function computes all the performance metrics at once 
get_performance_metrics<-function(actuals,test,model)
{
  predicted_response <- predict(model,newdata = test,type='response')
  predicted_probablity <- predict(model,newdata = test, type='prob')
    classes<-unique(predicted_response)
  
  if(length(classes)==2){
  #Accuracy
  accuracy <- length(predicted_response[predicted_response==actuals])/length(actuals)
  #Precision
  precision <- precision(data=predicted_response,reference=actuals,relevant=classes[1])
  #Recall
  recall <- recall(data=predicted_response,reference=actuals,relevant=classes[1])
  #Brier Score
  brier_score <- mean((predicted_probablity-(as.numeric(actuals)-1))^2)
  #AUC
  auc<-get_auc(actuals,predicted_probablity[,2])
  #F-Measure
  f_measure<-F_meas(data=predicted_response,reference = actuals)
  #Mathew's Correlation Co-efficient 
  mcc <- get_mcc(actuals,predicted_response,classes)
  
  return(c(accuracy,precision,recall,brier_score,auc,f_measure,mcc))
  }
  else{
    stop(paste0('We only support binary classifiers for now, the dependent variable has more than two outcome classes'))
  }
}


build_model<-function(classifier,train,...){
  
  if(classifier=='rf')
  {
  

    
    model<-drandomForest(response~., data=train, importance=TRUE, nExecutor = 5, ..., ntree=500, 
                         na.action=na.fail, trace=FALSE, 
                         completeModel=FALSE)
    class(model)<-c('randomForest.formula','randomForest')
    return(model)
    
  }
  
  if(classifier=='lrm')
  {
    model<-model<-train(response~.,data=train,method='glm',family='binomial')
    return(model)
    
  }
  
  if(classifier=='cart')
  {
    model<-train(response~.,data=train,method='rpart',tuneLength=10)
    return(model)
    
  }
  
  if(classifier=='knn')
  {
    model<-train(response~.,data=train,method='knn',preProcess=c('center','scale'))
    return(model)
    
  }
}


predict_generic<-function(model,test,type)
{
  if(type=='response')
  {
    return(predict(model,newdata=test,type='response')) 
  }
  if(type=='class')
  {
    return(predict(model,newdata=test,type='class')) 
  }
  if(type=='prob')
  {
    return(predict(model,newdata=test,type='prob'))
  }
  
}

remove_noise<-function(train,dep_var,cutpoint,target){
  
  return(train[(train[,dep_var]<cutpoint-target)|(train[,dep_var]>cutpoint+target),])
}

RWKH_framework<-function(classifier,data,parallel,n_cores,boot_size,dep_var,cutpoint,target,...)
{
  indices<-create_oosb_sample(data,boot_size)
  train_indices<-indices[[1]]
  test_indices<-indices[[2]]
  
  
  if(parallel<-TRUE){
    registerDoMC(n_cores)
    results<-foreach(i=1:boot_size,.combine=c,.multicombine=TRUE,.packages=c('randomForest.ddR','pROC','ROCR','caret'))%dopar%{
      
      train<-data[train_indices[[i]],]
      train<-remove_noise(train,dep_var,cutpoint,target)
      
      test<-data[test_indices[[i]],]
      
      train[,dep_var]<-NULL
      test[,dep_var]<-NULL
      #constructiong model
      model<-build_model(classifier,train)
      actuals<-test$response
      test$response<-NULL
      
      predicted<-predict_generic
      predicted<-predict(model,newdata=test,type='prob')
      
      list(list(get_performance_metrics(actuals,test,model),get_importance(model,colnames(test)),predict(model,newdata=test)))
      #list(list(get_auc(actuals,predicted),get_importance(model,colnames(test))))
    }
  return(results)
  }else{
  

    results<-list()
    for(i in 1:boot_size){
      
      
      train<-data[train_indices[[i]],]
      train<-remove_noise(train,dep_var,cutpoint,target)
      
      test<-data[test_indices[[i]],]
      
      train[,dep_var]<-NULL
      test[,dep_var]<-NULL
      
      #constructiong model
      model<-build_model(classifier,train)
      actuals<-test$response
      test$response<-NULL
      
      predicted<-predict_generic
      predicted<-predict(model,newdata=test,type='prob')
      
      results[[i]]<-list(get_performance_metrics(actuals,test,model),get_importance(model,colnames(test)))
      
    }
  
  return(results)
  }
}

#Analyzing discretization noise impact 

analyzeDiscretizationNoise<-function(data,dep_var,classifier,limit,step_size,
                                     parallel=FALSE,
                                     n_cores=1,boot_size=100,
                                     cutpoint=NULL,save_interim_results=FALSE,dest_path=NULL){
  



if(missing(dep_var)){
  stop('dependent variable column name needs to be sepcified')
}

if(missing(limit))
{
  stop('Limit value must be specified, so that the framework can demarcate the noisy area')
}

if(missing(step_size))
{
  stop('Choose a step size to form the increments to analyze the discretization noise in the noisy area')
}

if(parallel==TRUE)
{
  if(missing(n_cores)){
    stop('n_cores must be specified if you set the framework to run in parallel. *Highly advised as parallel excution will dramatically reduce runtime*')
  }
}

if(missing(boot_size))
{
  boot_size<-100
}
#Use the user provided cutpoint if provided. If not use median of the response variable as the cutpoint
if(is.null(cutpoint)){
  

    cutpoint<-median(data[,dep_var])
  
}else if(is.numeric(cutpoint)){
  cutpoint
}else{
  
  stop('The cutpoint needs to be numeric or specified as NULL, cannot have cutpoints of other classes')
}

if(!(classifier %in% c('rf','lrm','cart','knn')))
{
 stop('The framework only supports Radom Forest, Logistic Regression, CART and KNN classifiers, Other classifiers coming soon') 
}

if(save_interim_results==TRUE)
{
  if(missing(dest_path))
  {
    stop('To save the interim results, a destination path must be pro')
  }
}

sequence<-seq(0,limit,by=step_size)
print(dep_var)
print(cutpoint)
response<-ifelse(data[,dep_var]<=cutpoint,'class1','class2')
data<-cbind(data,response)

performance_results<-list()
importance_results<-list()

for(percentage in sequence)
{

  target<-cutpoint*(percentage/100)
  
  results<-RWKH_framework(classifier,data,parallel,n_cores,boot_size,dep_var,cutpoint,target)

  metrics<-do.call('rbind',lapply(results,function(x) x[[1]]))
  colnames(metrics)<-c('accuracy','precision','recall','brier_score','auc','f_measure','mcc')
  performance_results[[as.character(percentage)]]<-metrics
  
  imp<-do.call('rbind',lapply(results,function(x) x[[2]]))
  importance_results[[as.character(percentage)]]<-imp
 
}

if(save_interim_results==TRUE){
  saveRDS(performance_results,paste(dest_path,'/','performance.rds',sep=''))
  saveRDS(importance_results,paste(dest_path,'/','performance.rds',sep=''))
}

h<-head(sequence,1)
t<-tail(sequence,1)
stub1<-performance_results[[as.character(h)]]
stub2<-performance_results[[as.character(t)]]

for(k in 1:7){
print(paste(round(100-(median(stub1[,k])/median(stub2[,k]))*100,2),ifelse(wilcox.test(stub1[,k],stub2[,k],paired = FALSE)$p.value <0.05,'(S)','(NS)'),
      as.character(cohen.d(stub1[,k],stub2[,k])$magnitude),sep=' '))
}

}





