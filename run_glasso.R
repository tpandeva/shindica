# BSD 2-Clause License
#library(rslurm)
#library(readr)
#library(pracma)
#library(huge)
#library(psych)
get_stat=function(wi){
load("data/conf.RData")
    tp=NULL
    fp=NULL
    diag(wi)=0
    adj = wi
    adj=(adj+t(adj))/2
    thres = abs(unique(as.numeric(adj)))
    thres = thres[order(-thres)]

    t = thres[seq(min(100,length(thres)), length(thres),min(200,length(thres)))]
    t = unique(c(t,thres[length(thres)]))
    thres = t
    if(length(thres)<2) t=0

    for(t in thres){
      adj[abs(wi)>t]=1
      adj[abs(wi)<=t]=0
      tp = c(tp,sum(adj*conf)/2)
      fp = c(fp,sum(adj)/2-sum(adj*conf)/2)

    }
    dt = cbind.data.frame(thres,tp,fp)


    return(dt)
  }
get_fp_tp = function(lambda,params){


  S =params[[1]]
  n=params[[2]]
  d=ncol(S)
  load("data/conf.RData")

  mod0 = huge(S, lambda, method="glasso")
  wi = as.matrix(mod0$icov[[1]])
  wi=0.5*(abs(wi)+abs(t(wi)))
  loglik = NULL

  for(i in 1:length(mod0$icov)){
    K=mod0$icov[[i]]
    ch = chol(K)
    KS = K%*%S
    loglik = c(loglik, 2*sum(log(diag(ch)))-tr(KS) )
  }
  rm(K)
  rm(KS)
  rm(S)
  rm(ch)


  ebic = -n*0.5*loglik + log(n) * mod0$df +
    4 * 0.5 * log(d) * mod0$df

  dt = get_stat(wi)

  return(list(lambda,dt,ebic))
}
if(FALSE){


  file1 = "log/omics/S1.csv"
  file2 = "log/omics/S2.csv"

  X1 = read_csv(file1)
  X1[[1]]=NULL

  X2 = read_csv(file2)
  X2[[1]]=NULL

  X=cbind(X1,X2)
  S=cor(t(X))
  d = ncol(S)
  n=ncol(X)
  lambda.max = max(max(S - diag(d)), -min(S - diag(d)))
  lambda.min = 0.1 * lambda.max
  lambda = exp(seq(log(lambda.max), log(lambda.min), length = 60))

  pars = data.frame(lambda=rep(lambda,each=1))
  slurm = slurm_apply(f=get_fp_tp, params=pars, list(S,n),
                       jobname ="siica", nodes=2,
                       cpus_per_node=10, slurm_options=list(mem="100G"))



}