import numpy as np
import pandas as pd
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.9f}".format(x)})


def gaussian_kernel(distance, bandwidth):
 return (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)


def Mean_shift(init_mode,bandwidth):

    sum=0
    for i in range(len(X)):
        Dist=X[i]-init_mode
        Gau=gaussian_kernel(Dist,bandwidth)
        print("init_mode : ",init_mode,"Xi : ",X[i]," ,Weight : ",Gau)
        sum=sum+Gau


    #print("Sum : ",sum)

    sum1=0
    for i in range(len(X)):
        Dist=X[i]-init_mode
        Gau=gaussian_kernel(Dist,bandwidth)*X[i]
        #print("init_mode : ",init_mode,"Xi : ",X[i]," ,Weight * Xi : ",Gau)
        #print(Gau)
        sum1=sum1+Gau

    #print("Sum1 : ",sum1)
    print(sum)
    print(sum1)
    init_x=sum1/sum
    print("ㄴㅇㄴㅇㄹㄴㅇㄹㄴㅇㄹㄴㅇㄹㅇㄴㄹㄴㅇㄹㅇㄴㄹ", init_x)
    return init_x

X =[20,21,23,25,35,36,38,70,59,72]
X=np.array(X)
X=X.reshape(-1,1)
bandwidth=2.5

List=[]
a: int=0
for k in range(len(X)):
    init_x=X[k]
    for i in range(10):
        init_x=Mean_shift(init_x,bandwidth)

        a= a+1
        print("aaaaaaaaaa",a)
    List.append(init_x)

print("Final List: \n",List)

