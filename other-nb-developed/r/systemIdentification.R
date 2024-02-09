require(devtools)
install_github("surajy123/R-sysid")
require(sysid)

t=seq(0,10,.01)
y=sin(t)


dataMatrix <- matrix(rnorm(1000),ncol=5) 
data <- idframe(output=dataMatrix[,3:5],input=dataMatrix[,1:2],Ts=1)

data2 <- idframe(output=y, input=NULL, Ts=1)

iv(data)

plot(data2)
