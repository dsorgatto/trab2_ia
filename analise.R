pasta<-list.files()
pasta<-pastas[!grepl("\\.",pastas)]

for(i in 1:length(pasta)){
    temp<-read.csv(paste0(pasta[i],"/tabela.csv"))[,-1]
    temp<-data.frame(dataset=rep(pasta[i],30),
                     algoritimo=rep(c("knn","naive_bayes","decision_tree"),each=10),
                     acuracias=unlist(c(temp)),row.names=1:30)
    if(i==1){
        dados<-temp
    }else{
        dados<-rbind(dados,temp)
    }
}

#
tapply(dados$acuracias,list(dados$algoritimo),mean)
tapply(dados$acuracias,list(dados$algoritimo),sd)

#
tabela_mean<-tapply(dados$acuracias,list(dados$dataset,dados$algoritimo),mean)
tapply(dados$acuracias,list(dados$dataset,dados$algoritimo),sd)

tabela_mean
table(apply(tabela_mean,1,which.min))

