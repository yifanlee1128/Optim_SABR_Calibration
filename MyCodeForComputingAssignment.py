import numpy as np
import scipy.stats as stats
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
"""
initialize the origin data
"""
np.random.seed(3)
r_USD=0.022
r_BRL=0.065
S=3.724
table=[[1/365,20.98,1.2,0.15],
       [7/365,13.91,1.3,0.2],
       [14/365,13.75,1.4,0.2],
       [30/365,14.24,1.5,0.22],
       [60/365,13.84,1.75,0.27],
       [90/365,13.82,2.0,0.32],
       [180/365,13.82,2.4,0.43],
       [1,13.94,2.9,0.55]]

"""
present all results to avoid omission
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#r_d=r_BRL, r_f=r_USD

"""
define the function of F
"""
def getF(S,T,r_d,r_f):
    return S*np.exp((r_d-r_f)*T)

"""
implement SABR model using the Hagan et. al. approximation
"""
def getZ(K,F,a,v):
    return v/a*np.log(F/K)

def getX(Z,rho):
    return np.log((np.sqrt(1.0-2*rho*Z+Z**2)+Z-rho)/(1.0-rho))

def getHagenApproxVol(K,F,T,a,v,rho):
    Z=getZ(K,F,a,v)
    X=getX(Z,rho)
    return a*Z/X*(1+T*(rho*v*a/4.0+(2.0-3.0*rho**2)/24.0*v**2))

"""
function of delta with main inputs K and sigma
"""
def getDelta(K,F,sigma,T,r_f,phi):
    d2=np.log(F/K)/sigma/np.sqrt(T)-0.5*sigma*np.sqrt(T)
    return phi*np.exp(-r_f*T)*K/F*stats.norm.cdf(d2*phi)

"""
function of getting BS Model Price with main inputs K and sigma
"""
def getBSPrice(S,K,r_d,r_f,T,sigma):
    d1=(np.log(S/K)+(r_d-r_f+sigma**2/2.0)*T)/sigma/np.sqrt(T)
    d2=d1-sigma*np.sqrt(T)
    c=S*np.exp(-r_f*T)*stats.norm.cdf(d1)-K*np.exp(-r_d*T)*stats.norm.cdf(d2)
    p=K*np.exp(-r_d*T)*stats.norm.cdf(-d2)-S*np.exp(-r_f*T)*stats.norm.cdf(-d1)
    return c,p

"""
function of geting Price with main input K and corresponding caculated sigma with Hagan
"""
def getModelPrice(S,K,r_d,r_f,T,a,v,rho):
    F=getF(S,T,r_d,r_f)
    sigma=getHagenApproxVol(K,F,T,a,v,rho)
    return getBSPrice(S,K,r_d,r_f,T,sigma)

"""
The next three functions are for the optimization for K_ATM, K_RR_Call and K_RR_Put, K_BF_Call and K_BF_Put separately.
These three functions are only used for checking the implementation with the given example answer
The "getAll" function includes all of the implementations of these three functions for exhaustively optimizing
all unknown parameters for each tenor as required by the assignment.
"""
def getK_ATM(S,F,T,sigma,a,v,rho):
    f=lambda K: np.abs(getHagenApproxVol(K,F,T,a,v,rho)-sigma)
    res = optimize.basinhopping(f,x0=S,niter=5)
    return res.x[0]

def getK_RR(S,F,T,a,v,rho,RR,r_f):
    callSigma=lambda K_RR_Call:getHagenApproxVol(K_RR_Call,F,T,a,v,rho)
    putSigma=lambda K_RR_Put:getHagenApproxVol(K_RR_Put,F,T,a,v,rho)
    f1=lambda K_RR_Call,K_RR_Put:callSigma(K_RR_Call)-putSigma(K_RR_Put)-RR
    callDelta=lambda K_RR_Call:getDelta(K_RR_Call,F,callSigma(K_RR_Call),T,r_f,1)
    putDelta=lambda K_RR_Put:getDelta(K_RR_Put,F,putSigma(K_RR_Put),T,r_f,-1)
    f= lambda K_RR:np.abs(f1(K_RR[0],K_RR[1]))+np.abs(callDelta(K_RR[0])-0.25)+np.abs(putDelta(K_RR[1])+0.25)
    res=optimize.basinhopping(f,x0=np.array([S,S]),niter=5)
    return res.x[0],res.x[1]

def getK_BF(S,F,T,a,v,rho,r_f,r_d,sigma_ATM,sigma_BF):
    f1 = lambda K_BF_Call, K_BF_Put:getModelPrice(S,K_BF_Call,r_d,r_f,T,a,v,rho)[0]+getModelPrice(S,K_BF_Put,r_d,r_f,T,a,v,rho)[1]\
                                    -getBSPrice(S,K_BF_Call,r_d,r_f,T,sigma_ATM+sigma_BF)[0]-getBSPrice(S,K_BF_Put,r_d,r_f,T,sigma_ATM+sigma_BF)[1]
    callDelta = lambda K_BF_Call: getDelta(K_BF_Call, F, sigma_ATM+sigma_BF, T, r_f, 1)
    putDelta = lambda K_BF_Put: getDelta(K_BF_Put, F, sigma_ATM+sigma_BF, T, r_f, -1)
    f=lambda K_BF:np.abs(f1(K_BF[0],K_BF[1]))+np.abs(callDelta(K_BF[0])-0.25)+np.abs(putDelta(K_BF[1])+0.25)
    res=optimize.basinhopping(f,x0=np.array([S,S]),niter=5)
    return res.x[0],res.x[1]

"""
This function is used to calculate K with specified delta,such as 0.1 and -0.1 as the requirement
"""
def getKwithDelta(S,F,T,a,v,rho,delta,r_f,phi):
    if phi==1:
        callSigma = lambda K: getHagenApproxVol(K, F, T, a, v, rho)
        callDelta = lambda K: getDelta(K, F, callSigma(K), T, r_f, phi)
        f=lambda K: np.square(callDelta(K)-delta)
        res = optimize.basinhopping(f, x0=S+0.2, niter=1000,stepsize=0.0003)
        return res.x[0]
    elif phi==-1:
        putSigma = lambda K: getHagenApproxVol(K, F, T, a, v, rho)
        putDelta = lambda K: getDelta(K, F, putSigma(K), T, r_f, phi)
        f=lambda K:np.square(putDelta(K)-delta)
        res = optimize.basinhopping(f, x0=S-0.2, niter=1000,stepsize=0.0003)
        return res.x[0]
    else:
        raise ValueError("Phi must be either 1 or -1 !")


"""
This class is used to set boundaries for the optimizer
"""
lowerbound=[1.,1.,1.,1.,1.,0.01,0.01,-0.99]
upperbound=[10.,10.,10.,10.,10.,10.,10.,0.99]
class MyBounds(object):
    def __init__(self, xmax, xmin):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

myBound=MyBounds(upperbound,lowerbound)

"""
The main function to implement optimizations on required parameters
"""
def getAll(S,F,T,r_f,r_d,sigma_ATM,sigma_RR,sigma_BF,bound=myBound):
    call_RR_Sigma = lambda K_RR_Call,a,v,rho: getHagenApproxVol(K_RR_Call, F, T, a, v, rho)
    put_RR_Sigma = lambda K_RR_Put,a,v,rho: getHagenApproxVol(K_RR_Put, F, T, a, v, rho)
    f2 = lambda K_RR_Call, K_RR_Put,a,v,rho: call_RR_Sigma(K_RR_Call,a,v,rho) - put_RR_Sigma(K_RR_Put,a,v,rho) - sigma_RR

    callDelta_RR = lambda K_RR_Call,a,v,rho: getDelta(K_RR_Call, F, call_RR_Sigma(K_RR_Call,a,v,rho), T, r_f, 1)
    putDelta_RR = lambda K_RR_Put,a,v,rho: getDelta(K_RR_Put, F, put_RR_Sigma(K_RR_Put,a,v,rho), T, r_f, -1)
    f4 = lambda K_BF_Call, K_BF_Put,a,v,rho: getModelPrice(S, K_BF_Call, r_d, r_f, T, a, v, rho)[0] + \
                                     getModelPrice(S, K_BF_Put, r_d, r_f, T, a, v, rho)[1] \
                                     - getBSPrice(S, K_BF_Call, r_d, r_f, T, sigma_ATM + sigma_BF)[0] - \
                                     getBSPrice(S, K_BF_Put, r_d, r_f, T, sigma_ATM + sigma_BF)[1]

    callDelta_BF = lambda K_BF_Call: getDelta(K_BF_Call, F, sigma_ATM + sigma_BF, T, r_f, 1)
    putDelta_BF = lambda K_BF_Put: getDelta(K_BF_Put, F, sigma_ATM + sigma_BF, T, r_f, -1)

    f_all=lambda paras: np.square(getHagenApproxVol(paras[0], F, T, paras[5], paras[6], paras[7]) - sigma_ATM)+\
                        np.square(f2(paras[1], paras[2],paras[5],paras[6],paras[7])) + \
                        np.square(callDelta_RR(paras[1],paras[5],paras[6],paras[7]) - 0.25) + \
                        np.square(putDelta_RR(paras[2],paras[5],paras[6],paras[7]) + 0.25)+\
                        np.square(f4(paras[3], paras[4],paras[5],paras[6],paras[7])) + \
                        np.square(callDelta_BF(paras[3]) - 0.25) + \
                        np.square(putDelta_BF(paras[4]) + 0.25)

    minimizer_kwargs = {"method": "L-BFGS-B"}
    if np.allclose(T,1/365):
        iterNum=1000
    else:
        iterNum=100
    res = optimize.basinhopping(f_all, x0=[S+0.2,S+0.1,S-0.1,S+0.1,S-0.1,0.11,0.6,0.5], niter=iterNum,
                                accept_test=bound,stepsize=0.0005,minimizer_kwargs=minimizer_kwargs)
    print("Minimized error:",res.fun)
    print("Successfully find root:",res.message[0])
    return res.x

"""
do optimization
"""
results=[]
for i in range(len(table)):
    testData=table[i]
    print("-" * 90)
    print("T =",testData[0])
    result=getAll(S,getF(S,testData[0],r_BRL,r_USD),testData[0],r_USD,r_BRL,testData[1]/100,testData[2]/100,testData[3]/100)
    results.append(result)
    print("Optimized results:",result)
    print("-"*90)

"""
Calculate call strikes with +0.1 delta and put strikes with -0.1 delta with different tenors , then summarize all results
"""
Tenors=['ON','1W','2W','1M','2M','3M','6M','1Y']
result_table=pd.DataFrame()
for tenor,result,data in zip(Tenors,results,table):
    K_call_delta_plus10=getKwithDelta(S,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7],0.1,r_USD,1)
    K_put_delta_minus10=getKwithDelta(S,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7],-0.1,r_USD,-1)
    K_ATM=result[0]
    K_RR_Call=result[1]
    K_RR_Put=result[2]
    K_BF_Call=result[3]
    K_BF_Put=result[4]
    vol_ATM=getHagenApproxVol(K_ATM,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7])
    vol_RR_Call=getHagenApproxVol(K_RR_Call,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7])
    vol_RR_Put=getHagenApproxVol(K_RR_Put,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7])
    vol_BF_Call=getHagenApproxVol(K_BF_Call,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7])
    vol_BF_Put=getHagenApproxVol(K_BF_Put,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7])
    result_table[tenor]=[K_ATM,vol_ATM,K_RR_Call,vol_RR_Call,K_RR_Put,vol_RR_Put,K_BF_Call,vol_BF_Call,
                         K_BF_Put,vol_BF_Put,K_call_delta_plus10,K_put_delta_minus10,result[5],result[6],result[7]]

index=["K_ATM","vol_ATM","K_RR_Call","vol_RR_Call","K_RR_Put","vol_RR_Put","K_BF_Call","vol_BF_Call",
                         "K_BF_Put","vol_BF_Put","K_call_delta10","K_put_delta10","a","v","r"]
result_table.index=index
result_table=result_table.T

print(result_table)

"""
plot volatility cureves for each tenors
"""
K_min=min(result_table["K_put_delta10"].values)
K_max=min(result_table["K_call_delta10"].values)
KList=np.array(range(int(K_min*100-100),int(K_max*100+100)))/100

plt.figure(figsize=(10,6))
for result,data in zip(results,table):
    SigmaList=[getHagenApproxVol(k,getF(S,data[0],r_BRL,r_USD),data[0],result[5],result[6],result[7]) for k in KList]
    plt.plot(KList,SigmaList)
plt.ylim([0,0.3])
plt.legend(Tenors)
plt.title("Relationships between K and $\sigma$ for different tenors")
plt.xlabel("Strike price")
plt.ylabel("Implied volatility")
plt.savefig("Result2_VolatilityCurve.png")
plt.show()

result_table.to_csv("Result1_DataTable.csv")
