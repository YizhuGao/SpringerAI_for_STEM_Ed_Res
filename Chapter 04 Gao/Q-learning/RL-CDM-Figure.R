#Compare RL-CDM and random strategy
load("RL_CDM_tes_2.RData")
load("RL_CDM_base.RData")

iter <- c(1:n_iter)  

plot(iter, RL_att7, type="o", col="blue", lwd=2, pch=16,
     xlab="iter", ylab="total reward", main="RL-CDM V.S. Random strategy", ylim=c(0, 7))

lines(iter, RL_att7_base, type="o", col="orange", lwd=2, pch=16)

legend("topleft", legend=c("RL-CDM", "Random strategy"), col=c("blue", "orange"), lwd=2, pch=16)

#measurement error
load("RL_CDM_u0.05-0.1.RData")
error1<-RL_att7
mean(error1[50:60])
load("RL_CDM_u0.1-0.2.RData")
error2<-RL_att7
mean(error2[50:60])
load("RL_CDM_u0.2-0.3.RData")
error3<-RL_att7
mean(error3[50:60])
load("RL_CDM_u0.3-0.4.RData")
error4<-RL_att7
mean(error4[50:60])

iter <- c(1:n_iter)  

# figure
plot(iter, error1, type="o", col="#1F77B4", lwd=2, pch=16,
     xlab="iter", ylab="total reward", main="Different level of g s", ylim=c(0, 7))

# add lines
lines(iter, error2, type="o", col="#FF7F0E", lwd=2, pch=16)
lines(iter, error3, type="o", col="#2CA02C", lwd=2, pch=16)
lines(iter, error4, type="o", col="#D62728", lwd=2, pch=16)

legend("topleft", legend=c("g s~U(0.05,0.1)", "g s~U(0.1,0.2)","g s~U(0.2,0.3)","g s~U(0.3,0.4)"), col=c("#1F77B4", "#FF7F0E","#2CA02C","#D62728"), lwd=2, pch=16,inset=c(0.6, 0.6))
