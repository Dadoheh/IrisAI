class NaiveBayes:
        
    @staticmethod 
    def gauss(a1,classes,i): #a1, mean, std ----a1 to kolejny atrybut próbki sample
        exponent=np.exp(-(a1-classes[classes.keys()[i]])**2/(2*classes[classes.keys()[i+4]]**2)) #i = 1 -> sepal.length...
        return 1/(np.sqrt(2*np.pi*classes[classes.keys()[i+4]]))*exponent                       #i+4 = 5 -> std.sepal.length...
     
        
    @staticmethod
    def classify(trainSet,sample):
        #classes separation
        irisSetosa = iris.query('variety == "Setosa"')
        irisVirginica = iris.query("variety == 'Virginica'")
        irisVersicolor = iris.query("variety == 'Versicolor'")
        
        #mean and std vars and lists
        irisSetosaMeanStdGauss = irisSetosa.mean()
        irisVirginicaMeanStdGauss = irisVirginica.mean()
        irisVersicolorMeanStdGauss = irisVersicolor.mean()
        meanAndStdList = [irisSetosa, irisVirginica, irisVersicolor]
        gaussClassesList = [irisSetosaMeanStdGauss, irisVirginicaMeanStdGauss, irisVersicolorMeanStdGauss]
        
        #mean and std count
        for k in range(0,len(gaussClassesList)):
            for i in range(len(irisSetosa.keys())-1):
                    gaussClassesList[k]['std.{}'.format(
                        list(meanAndStdList[k].keys())[i])] = meanAndStdList[k][list(meanAndStdList[k].keys())[i]].std()        
        #gauss
        for k in range(0,len(gaussClassesList)):
            for i in range(len(irisSetosa.keys())-1):
                gaussClassesList[k]['gauss.{}'.format(
                    list(meanAndStdList[k].keys())[i])] = NaiveBayes.gauss(sample[i],gaussClassesList[k],i)         
                           
        #voting
        voting = {'setosa':0,'virginica':0,'versicolor':0}
        for k in range(0, len(gaussClassesList)):
            tmp = (gaussClassesList[k][8]*gaussClassesList[k][9]*gaussClassesList[k][10]*gaussClassesList[k][11])/3 #8-11
            voting[list(voting.keys())[k]] = tmp
            
            
        highest = max(voting.values())
        return ([k for k, v in voting.items() if v == highest])
            
        
#testing
sample = [0.583333,0.333333,0.779661,0.875000]
print("For given sample {} NaiveBayes returns: {}".format(sample,NaiveBayes.classify(iris,sample)))


