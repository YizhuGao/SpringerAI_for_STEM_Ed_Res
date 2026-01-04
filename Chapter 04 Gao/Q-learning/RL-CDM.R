setwd("C:\\Users\\TX\\Desktop")

#learning material
#possible knowledge point mastery state 
possible_state <- read.table("possible_states_KA.txt",head=F)

n_skill<-ncol(possible_state)
n_s <- nrow(possible_state)

#learning material
#transition probability matrix for 7 materials
#134
tes_134<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if(((possible_state[col,3]-possible_state[row,3])==1)&&(sum(abs(possible_state[col,c(1:2,4:7)]-possible_state[row,c(1:2,4:7)]))==0)){
      tes<-0.1
      tes_134[row,col]<-1-tes
      tes_134[row,row]<-tes
    }else if(((possible_state[col,4]-possible_state[row,4])==1)&&(sum(abs(possible_state[col,c(1:3,5:7)]-possible_state[row,c(1:3,5:7)]))==0)){
      tes<-0.1
      tes_134[row,col]<-1-tes
      tes_134[row,row]<-tes
    }else if(((possible_state[col,1]-possible_state[row,1])==1)&&(sum(abs(possible_state[col,c(2:7)]-possible_state[row,c(2:7)]))==0)){
      tes<-0.1
      tes_134[row,col]<-1-tes
      tes_134[row,row]<-tes
    }
  }
}
tes_134[9,]
#2
tes_2<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if((possible_state[col,2]-possible_state[row,2])==1){
      if(sum(abs(possible_state[col,c(1,3:7)]-possible_state[row,c(1,3:7)]))==0){
        tes<-0.1
        tes_2[row,col]<-1-tes
        tes_2[row,row]<-tes
      }
    }
  }
}
#3
tes_3<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if((possible_state[col,3]-possible_state[row,3])==1){
      if(sum(abs(possible_state[col,c(1:2,4:7)]-possible_state[row,c(1:2,4:7)]))==0){
        tes<-0.1
        tes_3[row,col]<-1-tes
        tes_3[row,row]<-tes
      }
    }
  }
}
#4
tes_4<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if((possible_state[col,4]-possible_state[row,4])==1){
      if(sum(abs(possible_state[col,c(1:3,5:7)]-possible_state[row,c(1:3,5:7)]))==0){
        tes<-0.1
        tes_4[row,col]<-1-tes
        tes_4[row,row]<-tes
      }
    }
  }
}
#57  
tes_57<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if(((possible_state[col,5]-possible_state[row,5])==1)&&(sum(abs(possible_state[col,c(1:4,6:7)]-possible_state[row,c(1:4,6:7)]))==0)){
      tes<-0.1
      tes_57[row,col]<-1-tes
      tes_57[row,row]<-tes
    }else if(((possible_state[col,7]-possible_state[row,7])==1)&&(sum(abs(possible_state[col,c(1:6)]-possible_state[row,c(1:6)]))==0)){
      tes<-0.1
      tes_57[row,col]<-1-tes
      tes_57[row,row]<-tes
    }
  }
}
#6
tes_6<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if((possible_state[col,6]-possible_state[row,6])==1){
      if(sum(abs(possible_state[col,c(1:5,7)]-possible_state[row,c(1:5,7)]))==0){
        tes<-0.1
        tes_6[row,col]<-1-tes
        tes_6[row,row]<-tes
      }
    }
  }
}
#7
tes_7<-diag(31)
for(row in 1:31){
  for(col in 1:31){
    if((possible_state[col,7]-possible_state[row,7])==1){
      if(sum(abs(possible_state[col,c(1:6)]-possible_state[row,c(1:6)]))==0){
        tes<-0.1
        tes_7[row,col]<-1-tes
        tes_7[row,row]<-tes
      }
    }
  }
}
n_material <- 7
trans_mat<-array(0,dim=c(7,n_s,n_s))

trans_mat[1,,]<-tes_134
trans_mat[2,,]<-tes_2
trans_mat[3,,]<-tes_3
trans_mat[4,,]<-tes_4
trans_mat[5,,]<-tes_57
trans_mat[6,,]<-tes_6
trans_mat[7,,]<-tes_7

#learning target
#total learning step
n_epoch <- 7

#measurement model
set.seed(123)
#library("GDINA")
n_item<-n_skill*(n_epoch+1)
attr_mode<-att.structure(NULL,n_skill)$att.st[2:(n_skill+1),]
bind<-function(x,t){
  m<-x
  for(i in 1:(t-1)){
    x<-rbind(x,m)
  }
  return(x)
}

Q_item<-bind(attr_mode,(n_epoch+1))

item_g<-runif(n_item,0.05,0.2)
item_s<-runif(n_item,0.05,0.2)

#learner
n_sample <- 20
n_iter<-60


gamma1<-10
gamma2<-0.01

#because of intercept
n_para<-n_skill+1


RL_att7<-c(0)


beta<-array(0,dim=c(n_s,n_epoch,n_material,n_para))
ini_state<-c(0)
for(i in 1:n_sample){
  #ini_state is (0,0,0,0,0,0,0) the number is 1
  ini_state[i]<-1
  }
key<-0
for(iter in 1:n_iter){
  beta0<-beta
  gene_data<-generate_data(n_sample,n_epoch,ini_state,n_material,n_s,n_skill,possible_state,n_item,Q_item,item_g,item_s,beta,gamma1,n_para,trans_mat)
  beta<-updata_paras(gene_data,n_sample,n_epoch,n_skill,n_s,n_para,n_material,beta,possible_state,gamma2)
  if(mean(gene_data[[1]][5,,1])>mean(key)){
    beta_RL<-beta
  } 
  key<-gene_data[[1]][5,,1]
  RL_att7[iter]<-mean(key)
}


RL_att7_beta<-beta
RL_att7_mosbeta<-beta_RL

save.image("C:/Users/TX/Desktop/RL_CDM_tes.RData")


RL_att7_base<-c(0)
beta<-array(0,dim=c(n_s,n_epoch,n_material,n_para))
ini_state<-c(0)
for(i in 1:n_sample){
  #ini_state is (0,0,0,0,0,0,0) the number is 1
  ini_state[i]<-1
}
key<-0
for(iter in 1:n_iter){
  gene_data<-generate_data(n_sample,n_epoch,ini_state,n_material,n_s,n_skill,possible_state,n_item,Q_item,item_g,item_s,beta,gamma1,n_para,trans_mat)
  key<-gene_data[[1]][5,,1]
  RL_att7_base[iter]<-mean(key)
}

save.image("C:/Users/TX/Desktop/RL_CDM_base.RData")





#RL-CDM code

make_prob<-function(post){
  res<-0
  max_post<-post[which.max(post)]
  x<-post
  for(i in 1:length(post)){
    x[i]=exp(x[i]-max_post)
    res=res+x[i]
  }
  for(j in 1:length(x)){
    x[i]=x[i]/res
  }
  return(x)
}

get_mle_posts<-function(index_state_true,possible_states_KA,n_item,Q_item,item_g,item_s,i_epoch){
  n_s<-nrow(possible_states_KA)
  n_skill<-ncol(possible_states_KA)
  state_true<-t(possible_states_KA[index_state_true,])
  l=1+i_epoch*n_skill
  r=i_epoch*n_skill+n_skill
  Q_item_choose<-Q_item[l:r,]
  item_s_choose<-item_s[l:r]
  item_g_choose<-item_g[l:r]
  n_item_choose<-n_s-1
  ideal_resp<-rep(1,n_item_choose)
  for(i in 1:nrow(Q_item_choose)){
    answer=state_true-Q_item_choose[i,]
    for(j in 1:ncol(Q_item_choose)){
      if(answer[j]<0){
        ideal_resp[i]=0
      }else if(is.na(answer[j])) {
        print(i_epoch)
      }
    }
  }
  true_resp_prob<-((1-item_s_choose)*ideal_resp)+(item_g_choose*(1-ideal_resp))
  true_resp<-as.numeric(true_resp_prob>runif(n_item_choose,0,1))
  posts<-matrix(0,n_s,1)
  special_prob<-matrix(1,n_s,1)
  for(index_state in 1:n_s){
    state_tes<-t(t(possible_states_KA[index_state,]))
    ideal_resp_tes<-rep(1,n_item_choose)
    for(i in 1:nrow(Q_item_choose)){
      answer=state_tes-Q_item_choose[i,]
      for(j in 1:ncol(Q_item_choose)){
        if(answer[j]<0){
          ideal_resp_tes[i]=0
        }else if(is.na(answer[j]) ) {
          print(i_epoch)
        }
      }
    }
    true_resp_prob_tes<-((1-item_s_choose)*ideal_resp_tes)+(item_g_choose*(1-ideal_resp_tes))
    mle<-item_g_choose^(1-ideal_resp_tes)*(1-item_s_choose)^(ideal_resp_tes)
    for(i in 1:n_item_choose){
      if(true_resp[i]==1){
        special_prob[index_state]<-special_prob[index_state]*true_resp_prob_tes[i]
        posts[index_state]=posts[index_state]+(true_resp_prob_tes[i])
      }else{
        special_prob[index_state]<-special_prob[index_state]*(1-true_resp_prob_tes[i])
        posts[index_state]=posts[index_state]+(1-true_resp_prob_tes[i])
      }
    }
  }
  post<-posts/sum(posts)
  return(post)
}

get_f<-function(post,n_s,n_para,n_skill,possible_states_KA){
  f<-c(0)
  f[1]<-1
  i=2
  for(k in 1:n_skill){
    f[i]<-0
    for(s in 1:n_s){
      if(possible_states_KA[s,k]==1){
        f[i]=f[i]+post[s]
      }
    }
    f[i]=f[i]
    
    i=i+1
  }
  f[1]=1
  if((i-1)!=n_para){
    print("wrong in get_f")
  }
  return(f)
}

calculate_q<-function(post,beta,n_para,n_s,n_skill,possible_states_KA){
  q<-0
  f<-get_f(post,n_s,n_para,n_skill,possible_states_KA)
  for(i in 1:(n_para)){
    q=q+f[i]*beta[i]
  }
  return(q)
}

choose_material<-function(post,beta,gamma1,Pmat_total,n_s,n_skill,possible_states_KA){
  n_material<-length(Pmat_total)/(n_s*n_s)
  n_para<-n_skill+1
  prob<-c(0)
  for(index_d in 1:n_material){
    prob[index_d]<-exp(gamma1*calculate_q(post,beta[index_d,],n_para,n_s,n_skill,possible_states_KA))
    if(prob[index_d]==Inf){
      prob[index_d]=10000
    }
  }
  chosen_d<-sample(1:n_material,1,replace=TRUE,prob=prob)
  return(chosen_d)
}


state_transit<-function(curr_s,Pmat_KA,curr_d,n_s){
  next_true_state<-sample(1:n_s,1,prob=Pmat_KA[curr_d,curr_s,])
  return(next_true_state)
}

calculate_reward<-function(sample,index_epoch,n_skill,n_s,possible_states_KA,t_total,curr_d){
  reward<-0
  init_s<-sample[index_epoch]
  final_s<-sample[index_epoch+1]
  for(index_skill in 1:n_skill){
    reward=reward+(possible_states_KA[final_s,index_skill]-possible_states_KA[init_s,index_skill])
  }
  return(reward)
}

calculate_total_reward<-function(sample,index_sample,n_skill,n_epoch,possible_states_KA){
  total_reward<-0
  
  for(s in 1:n_epoch){
    for(index_skill in 1:n_skill){
      init_s<-sample[s]
      final_s<-sample[s+1]
      total_reward=total_reward+(possible_states_KA[final_s,index_skill]-possible_states_KA[init_s,index_skill])
    }
  }
  return(total_reward)
}


generate_data<-function(n_sample,n_epoch,ini_state,n_material,n_s,n_skill,possible_states_KA,n_item,Q_item,item_g,item_s,beta,gamma1,n_para,trans_mat){
  Pmat_total<-trans_mat
  sample<-array(0,dim=c(5,n_sample,n_epoch+1))
  posts<-array(0,dim=c(n_sample,n_epoch+1,n_s))
  for(index_sample in 1:n_sample){
    curr_s<-ini_state[index_sample]
    sample[1,index_sample,1]<-curr_s
    i_epoch=0
    posts[index_sample,1,]<-get_mle_posts(curr_s,possible_states_KA,n_item,Q_item,item_g,item_s,i_epoch)
    sample[2,index_sample,1]<-which.max(posts[index_sample,1,])
    for(index_epoch in 1:n_epoch){
      index_s0<-sample[2,index_sample,index_epoch]
      curr_d<-choose_material(posts[index_sample,index_epoch,],beta[index_s0,index_epoch,,],gamma1,Pmat_total,n_s,n_skill,possible_states_KA)
      sample[3,index_sample,index_epoch]<-curr_d
      next_true_state<-state_transit(curr_s,Pmat_total,curr_d,n_s)
      curr_s<-next_true_state
      sample[1,index_sample,index_epoch+1]<-curr_s
      posts[index_sample,index_epoch+1,]<-get_mle_posts(curr_s,possible_states_KA,n_item,Q_item,item_g,item_s,index_epoch)
      sample[2,index_sample,index_epoch+1]<-which.max(posts[index_sample,index_epoch+1,])
      reward<-calculate_reward(sample[1,index_sample,],index_epoch,n_skill,n_s,possible_states_KA,t_total,sample[3,index_sample,index_epoch])
      sample[4,index_sample,index_epoch]=as.numeric(reward)
          }
    total_reward<-calculate_total_reward(sample[1,index_sample,],index_sample,n_skill,n_epoch,possible_states_KA)
    sample[5,index_sample,1]<-total_reward
    
  }
  gene_data<-list(sample,posts)
  return(gene_data)
}

updata_paras<-function(gene_data,n_sample,n_epoch,n_skill,n_s,n_para,n_material,beta,possible_states_KA,gamma2){
  sample<-gene_data[[1]]
  posts<-gene_data[[2]]
  
  delta_beta<-array(0,dim=c(n_s,n_epoch,n_material,n_para))
  for(index_sample in 1:n_sample){
    for(index_epoch in 1:n_epoch){
      index_s0=sample[2,index_sample,index_epoch]
      curr_d=sample[3,index_sample,index_epoch]
      f<-get_f(posts[index_sample,index_epoch,],n_s,n_para,n_skill,possible_states_KA)
      curr_q<-calculate_q(posts[index_sample,index_epoch,],beta[index_s0,index_epoch,curr_d,],n_para,n_s,n_skill,possible_states_KA)
      if(index_epoch==n_epoch){
        q_max=0
      }else{
        q<-c(0)
        for(index_d in 1:n_material){
          q[index_d]<-calculate_q(posts[index_sample,index_epoch+1,],beta[index_s0,index_epoch+1,index_d,],n_para,n_s,n_skill,possible_states_KA)
          q_max<-max(q)
        }
      }
      curr_G<-sample[4,index_sample,index_epoch]+q_max
      for(index_para in 1:n_para){
        delta_beta[index_s0,index_epoch,curr_d,index_para]=delta_beta[index_s0,index_epoch,curr_d,index_para]+(curr_G-curr_q)*f[index_para]
      }
    }
  }
  for(index_s0 in 1:n_s){
    for(index_d in 1:n_material){
      for(index_para in 1:n_para){
        for(index_epoch in 1:n_epoch){
          beta[index_s0,index_epoch,index_d,index_para]=beta[index_s0,index_epoch,index_d,index_para]+delta_beta[index_s0,index_epoch,index_d,index_para]*gamma2
        }}
    }
  }
  return(beta)
}




