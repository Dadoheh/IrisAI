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


#soft harvest
class SoftSetegIris:
    @staticmethod
    def classifyIris(sample, irisFrame):
        listOfVariety,listOfColumns = [],[]
        dictOfVariety, dictOfColumns = {},{}
        for i in range(len(irisFrame['variety'])):
            if irisFrame['variety'][i] not in listOfVariety:
                listOfVariety.append(irisFrame['variety'][i])
        for i in range(len(listOfVariety)):
            dictOfVariety[listOfVariety[i]] = {}
        listOfColumns = pandas.Index.to_list(irisFrame.columns)
        for i in range(len(listOfColumns)-1):
            dictOfColumns[listOfColumns[i]] = 0
        for i in range(len(listOfVariety)):
            dictOfVariety[listOfVariety[i]] = 0
        dictOfDicts = dict.fromkeys(listOfVariety, dictOfColumns)
        mainSoftSet = pandas.concat({k: pandas.DataFrame.from_dict(v, 'index') for k, v in dictOfDicts.items()},axis=1) 
        for i in range(len(dictOfColumns)-1):
            dictOfColumns[listOfColumns[i]] = irisFrame[listOfColumns[i]].mean() 
            temp = irisFrame.loc[irisFrame['variety']== listOfVariety[i]]
            for j in range(len(dictOfColumns)):
                if temp[listOfColumns[j]].mean() > dictOfColumns[listOfColumns[j]]: 
                   mainSoftSet.loc[listOfColumns[j],listOfVariety[i]] = 1 
                elif temp[listOfColumns[j]].mean() < dictOfColumns[listOfColumns[j]]: 
                    mainSoftSet.loc[listOfColumns[j],listOfVariety[i]] = 0


        return mainSoftSet
