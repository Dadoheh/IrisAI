import numpy as np
import random
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns 
import math
sns.set_palette('husl')

iris = pandas.read_csv('iris_.csv') 
sns.pairplot(iris,hue='variety',markers='+') 
sns.violinplot(y='variety',x='sepal.length',data=iris,inner='quartile')


#normalization
class DataProcessing: 
    @staticmethod
    def shuffle(x): 
        for i in range(len(x)-1,0,-1):
            j = random.randint(0,i)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i] 
        return x
    
    @staticmethod
    def splitSet(x):
        train = x.head(round((0.7)*len(x)))
        val = x.loc[round((0.7)*len(x)):len(x)+1]
        return train, val
      
    @staticmethod
    def normalize(x):
        listaRekordów = x.to_dict('list')
        listaTytułów = x.columns.tolist()  
        y = {} 
        listaUnsorted = listaRekordów.copy()
        for i in range(0,len(listaTytułów)-1):
            listaSorted = listaRekordów[listaTytułów[i]]
            minim = listaSorted[0]
            maxim = listaSorted[0]
            for k in range(len(listaSorted)):
                temp = listaSorted[k]
                if temp>maxim:
                    maxim = temp
                elif temp<minim:
                    minim = temp
            tempLista = []
            for j in range(0,150):
                tempVar = listaUnsorted[listaTytułów[i]][j]
                tempVar = (tempVar-minim)/(maxim-minim)
                tempLista.append(tempVar)
            y[str(listaTytułów[i])] = tempLista
        y['variety'] = listaUnsorted['variety']
        y = pandas.DataFrame.from_dict(y)
        return y


#clustering
class KNN:
    @staticmethod
    def minkowskiMetric(v1,v2,m): 
        distance = 0
        for i in range(len(v1)-2):
            distance+=abs(v1[i]-v2[i])**m
        distance = distance**(1/m)
        return distance
    
    @staticmethod
    def clustering(testSample,x,k,classes):
        distances = []
        for i in x.index:
            distances.append((KNN.minkowskiMetric(testSample,x.iloc[i],2),i))
        n = len(distances)         
        distances.sort()
    
        #glosowanie
        for i in range(0,k):
            classes[x.iloc[distances[i][1]].variety]+=1       
        return max(classes, key=classes.get)


#softset
class SoftSetIris:
    @staticmethod
    def classifyIris(sample, irisTrainBase, valBase):
        
        irisTrainGroupMean = irisTrainBase.groupby('variety').mean()
        print("\n irisTrainGroupMean = Zbiór treningowy z użyciem średnich wartości \n{}\n".format(irisTrain.groupby('variety').mean()))
        irisTrainMean = irisTrainBase.mean()
        print("\n irisTrainMean = Zbiór treningowy z użyciem średnich wartości dla całej klasy\n{}\n".format(irisTrain.mean()))
        print("Przykładowy rekord irisTrainGroupMean {} ".format(irisTrainGroupMean['sepal.length']['Setosa'])) #pojedynczy rekord 
        print("Przykładowy rekord irisTrainMean {} ".format(irisTrainMean['sepal.length']))
        nullSet = irisTrainGroupMean.copy(deep=True)
        
        #setting 0,1
        for i in range(len(irisTrainGroupMean)):
            for j in range(len(irisTrainMean)):
                if irisTrainGroupMean.iloc[i][j] > irisTrainMean.iloc[j]:
                    nullSet.iloc[i][j] = 1
                if irisTrainGroupMean.iloc[i][j] < irisTrainMean.iloc[j]:
                    nullSet.iloc[i][j] = 0
        print("\n Zbiór miękki irysów \n{}\n".format(nullSet))

        #typing class of Iris
        equal = {}
        namesOfClasses = list(nullSet.index)
        for i in range(0,3):
            tempEqual = 0
            for j in range(0,4):
                tempEqual += sample[j]*nullSet.iloc[i][j]
            equal[namesOfClasses[i]] = tempEqual
        highest = max(equal.values())
        return ([k for k, v in equal.items() if v == highest])
        
        return equal

    
#Clustering Testing
corrected=0
k=5   
irisShuffled = DataProcessing.shuffle(iris)
irisNormalized = DataProcessing.normalize(iris)
irisMixedTrain, irisMixedVal = DataProcessing.splitSet(iris)
for i in range(len(irisMixedVal.index)):
    classes={'Setosa':0,'Virginica':0,'Versicolor':0}
    result = KNN.clustering(irisMixedVal.iloc[i],irisMixedTrain,k,classes)
    if result == irisMixedVal.iloc[i].variety:
        corrected+=1
print(corrected)
accurancy = corrected/len(irisMixedVal)*100
print("Celność AI metodą klasteryzacji KNN: {}%".format(round(accurancy,2)))
 
#SoftSet Testing
iris=DataProcessing.shuffle(iris)
iris=DataProcessing.normalize(iris)
irisTrain, irisVal = DataProcessing.splitSet(iris)
sample = irisTrain.iloc[random.randrange(0,len(irisTrain))]
sampleDict = sample.to_dict()
print("AI oceniło klasę próbki jako: ===== {}\n".format(SoftSetIris.classifyIris(sample, irisTrain, irisVal)))
print("Rzeczywista klasa próbki: ===== {}\n".format(sample.loc['variety']))

    
