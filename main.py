import numpy as np
import pandas as pd
np.random.seed(42)
#from sklearn.datasets import load_iris, load_digits,fetch_california_housing

#import kagglehub
#from kagglehub import KaggleDatasetAdapter



df = pd.read_csv("data/realestate_data_southcarolina_2025.csv")
df = pd.get_dummies(df, columns=['type', 'sub_type'], drop_first=True)
df = df.dropna()
df = df.astype(np.float64)
print(df.columns)
print(df.shape)

data = df.to_numpy()
print(data[1])
dataFeatures=data[:,:5]
print(dataFeatures.shape)
dataLabels=data[:,-1:]

mean = dataFeatures.mean(axis=0)       # mean of each column
std = dataFeatures.std(axis=0)         # std of each column
X_standardized = (dataFeatures - mean) / std

y = dataLabels  # shape (2724, 1)
y_mean = y.mean()
y_std = y.std()
y_standardized = (y - y_mean) / y_std



class Activation:
    def __init__(self,type):
        self.type=type

    def activate(self):
        if self.type=='lin':
            self.z=(self.linkToLayer.data@self.linkToLayer.weights)+self.linkToLayer.bias
            self.output=self.z
            self.activationDer = np.ones_like(self.output)

        elif self.type=='rel':    
            self.z=(self.linkToLayer.data@self.linkToLayer.weights)+self.linkToLayer.bias
            self.z = np.clip(self.z, -500, 500)
            self.output=np.maximum(0,self.z).astype(float)
            self.activationDer=(self.z>0).astype(float)

        elif self.type=='sig':        
            self.z=(self.linkToLayer.data@self.linkToLayer.weights)+self.linkToLayer.bias
            self.output=1/(1+np.exp(-self.z))
            self.activationDer=self.output*(1-self.output)

        elif self.type=='smax':   
            self.z=self.linkToLayer.data
            self.z2=self.z-np.max(self.z,axis=1,keepdims=True)
            exp=np.exp(self.z2)
            expSum=np.sum(exp,axis=1,keepdims=True)
            self.output=exp/expSum
            jacobians = []
            for i in range(self.output.shape[0]):
                s = self.output[i]
                J1=np.diag(s)
                s=s.reshape(-1,1)
                J2=s@s.T
                J0=J1-J2
                jacobians.append(J0)
            jacobian = np.array(jacobians)
            self.activationDer = jacobian

    def backPropogate(self,epochs):     
        batch_size=20
        total_samples = NN.head.data.shape[0]
        for epoch in range(epochs-1):
            print((epoch/epochs)*100)
            for batchStart in range(0,total_samples,batch_size):
                batchStop=min(batchStart+batch_size,total_samples)
                batchIndices=np.arange(batchStart,batchStop)
                yBatch=NN.head.labels[batchIndices]
                yhatBatch=NN.tail.activation.output[batchIndices]

                if NN.tail.type in ['smax','lin']:
                    NN.delta=yhatBatch-yBatch
                elif NN.tail.type =='sig':
                    NN.delta=(yhatBatch-yBatch)/(yhatBatch*(1-yhatBatch))
                backPropogateLayerRef=NN.tail
                while backPropogateLayerRef:
                    if backPropogateLayerRef.type in ['lin','sig','rel']:
                        NN.delta=NN.delta*backPropogateLayerRef.activation.activationDer[batchIndices]
                    elif backPropogateLayerRef.type=='smax':
                        updatedDelta=[]
                        for i in range(NN.delta.shape[0]):
                            subDelta=NN.delta[i].reshape(-1,1)
                            subJacobian=backPropogateLayerRef.activation.activationDer[batchIndices]
                            subJ=subJacobian[i]
                            calculation=subJ@subDelta
                            updatedDelta.append(calculation)
                        NN.delta=np.array(updatedDelta)    
                    backPropogateLayerRef.derw=backPropogateLayerRef.data[batchIndices].T@NN.delta
                    backPropogateLayerRef.derb = np.sum(NN.delta, axis=0, keepdims=True)
                    if backPropogateLayerRef.prev is not None:
                        NN.delta = NN.delta @ backPropogateLayerRef.weights.T
                    backPropogateLayerRef=backPropogateLayerRef.prev

                weightsReCal=NN.tail
                while weightsReCal:
                    if weightsReCal.type!='smax':
                        weightsReCal.weights=weightsReCal.weights-(0.0001*weightsReCal.derw)
                        weightsReCal.bias=weightsReCal.bias-(0.0001*weightsReCal.derb)
                    weightsReCal=weightsReCal.prev

                activateOutputs=NN.head
                while activateOutputs:
                    activateOutputs.activation.activate()
                    if activateOutputs.next:
                        activateOutputs.next.data=activateOutputs.activation.output
                    activateOutputs=activateOutputs.next            
                NN.delta=None

class NN:   
    labels=None
    layers=None
    head=None
    tail=None
    cur=None
    finalWeights=None

    def addFirstLayer(self,data,labels,type,neurons):
        NN.head=self
        NN.labels=np.array(labels)
        NN.cur=self
        self.prev=None
        self.next=None
        self.data=np.array(data)
        self.type=type
        self.weights = np.random.randn(self.data.shape[1],neurons)
        self.bias=np.zeros((1,neurons))
        self.activation=Activation(type)
        self.activation.linkToLayer=self
        self.activation.activate()
        

    def addHiddenLayer(self,type,neurons):
        NN.cur.next=self
        self.prev=NN.cur
        NN.cur=self
        self.next=None
        self.data=self.prev.activation.output
        self.type=type
        if not type=='smax':
            self.weights = np.random.randn(self.data.shape[1],neurons)
            self.bias=np.zeros((1,neurons))
            self.activation=Activation(type)
            self.activation.linkToLayer=self
            self.activation.activate()
        else:
            print("You are trying to use softmax in the hidden layer. This is not allowed")
            
        

    def addFinalLayer(self,type,neurons,epochs):
        NN.cur.next=self
        NN.tail=self
        self.prev=NN.cur
        NN.cur=self
        self.next=None
        NN.cur.next=None
        self.data=self.prev.activation.output
        self.type=type
        if self.type=='smax':
            self.weights=np.ones((self.data.shape[1],neurons))
            self.bias=np.zeros((1,neurons))
            if not neurons==self.prev.activation.output.shape[1]:
                print("You are trying to use softmax in the final layer. The input of softmax should be equal to the neurons specified in finalLayer. Input is: ",self.prev.activation.output[0], "neurons is: ",neurons)
        else:
            self.weights = np.random.randn(self.data.shape[1],neurons)
            self.bias=np.zeros((1,neurons))
        self.activation=Activation(type)    
        self.activation.linkToLayer=self
        self.activation.activate()
        self.activation.backPropogate(epochs)


    def predict(self,data,layer):   
        a=layer
        if a.type=='lin':
            z=(data@a.weights)+a.bias
            f=z
        elif a.type=='rel':
            z=(data@a.weights)+a.bias
            f=np.maximum(0,z)
        elif a.type=='sig':
            z=(data@a.weights)+a.bias
            z = np.clip(z, -500, 500) 
            f=1/(1+np.exp(-z))    
        elif a.type == 'smax':
            z=data
            z = z - np.max(z, axis=1, keepdims=True)  #This is for stabilization, if we reduce a constant from a list of numbers, the smax outcome wont change
            num = np.exp(z) 
            sumNum = np.sum(num, axis=1, keepdims=True) #keepdims is for broadcasting 
            f = num / sumNum
        if a.next:
            self.predict(f,a.next)
        else:
            print("predicted value is: ",f)







l1=NN()
l1.addFirstLayer(X_standardized,y_standardized,'lin',64)
l2=NN()
l2.addHiddenLayer('sig',64)
l3=NN()
l3.addHiddenLayer('rel',32)
l4=NN()
l4.addHiddenLayer('sig',16)
l6=NN()
l6.addHiddenLayer('rel',8)
l5=NN()
l5.addFinalLayer('lin',1,1000)



d = X_standardized[20].reshape(1, -1)  # shape (1, 5)
yd=y_standardized[20]

l5.predict(d,NN.head)

print("actual value is ",yd)