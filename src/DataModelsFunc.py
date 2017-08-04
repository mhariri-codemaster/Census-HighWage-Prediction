import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

classes = np.array([0,1])

def PrintScores(scores,TF):
    print('\n\nScores:')
    print('------------------------')
    for scr in scores:
        print(scr + ' Classifier: ' + str(scores[scr]))
        print('Correct Label 1 Rate: ' + str(TF[scr][0]))
        print('Correct Label 0 Rate: ' + str(TF[scr][1]))
        print('\n')
        
def PlotEpochScores(epoch_scores,scores):
    # Print scores vs each epoch
    for model in epoch_scores:
        scrs = epoch_scores[model]
        LE = len(scrs)
        plt.plot(range(LE),scrs,label=model)
    plt.plot(range(LE),scores['BaseLine']*np.ones((LE,1)))
    plt.legend(loc='best')
    plt.show()

def BaselineScore(src_path,ch_sz,labels):
    # Baseline is chosen to be that all predictions are 0
    FullX = pd.read_csv(src_path,chunksize=ch_sz)
    score = 0
    length = 0
    for X in FullX:
        score += sum(X[labels])
        length += len(X)
    score = 1.0-float(score)/float(length)
    return score, (0.0,1.0)
        
def ModelFit(model,src_path,ch_sz,epochs,features,labels,SS):           
    # Training model with data as is
    epoch_scores = []
    for ep in range(epochs):
        FullX = pd.read_csv(src_path,chunksize=ch_sz)
        for X in FullX:
            SS.transform(X[features])
            model.partial_fit(X[features],X[labels],classes=classes)
        print('Epoch ' + str(ep+1) + ' completed')
        epoch_scores.append(ModelTestScore(model,src_path,ch_sz,features,labels,SS))
    return epoch_scores
    
def ModelFitOverSample(model,src_path,ch_sz,epochs,features,labels,SS):           
    # Training model with oversample of minority class
    epoch_scores = []
    for ep in range(epochs):
        FullX = pd.read_csv(src_path,chunksize=ch_sz)
        for X in FullX:
            SS.transform(X[features])
            undersampled = X.loc[X[labels]==1]
            repeat = 4
            X2 = np.tile(undersampled,(repeat,1))
            X2 = pd.DataFrame(X2,columns=X.columns)
            X = X.append(X2,ignore_index=True)
            X = X.sample(frac=1).reset_index(drop=True)
            model.partial_fit(X[features],X[labels],classes=classes)
        print('Epoch ' + str(ep+1) + ' completed')
        epoch_scores.append(ModelTestScore(model,src_path,ch_sz,features,labels,SS))
    return epoch_scores
    
def ModelFitUnderSample(model,src_path,ch_sz,epochs,features,labels,SS):           
    # Training model with undersample of majority class
    epoch_scores = []
    for ep in range(epochs):
        FullX = pd.read_csv(src_path,chunksize=ch_sz)
        for X in FullX:
            SS.transform(X[features])
            oversampled = X.loc[X[labels]==0]
            frac2leave = 0.5
            oversampled = oversampled.sample(frac=frac2leave)
            X.drop(X.index[X[labels]==0],inplace=True)
            X = X.append(oversampled,ignore_index=True)
            X = X.sample(frac=1).reset_index(drop=True)
            model.partial_fit(X[features],X[labels],classes=classes)
        print('Epoch ' + str(ep+1) + ' completed')
        epoch_scores.append(ModelTestScore(model,src_path,ch_sz,features,labels,SS))
    return epoch_scores
    
def ModelTestScore(model,src_path,ch_sz,features,labels,SS): 
    # Assess model on all data          
    score = []
    FullX = pd.read_csv(src_path,chunksize=ch_sz)
    for X in FullX:
        SS.transform(X[features])
        score.append(model.score(X[features],X[labels]))       
    return np.mean(score)
    
def ModelTestFalseTrue(model,src_path,ch_sz,features,labels,SS):
    # Assess model on majority and minority separately
    TP = 0 # True Positives
    TN = 0 # True Negatives
    TNP = 0 # Total Number of positive labels
    TNN = 0 # Total Number of negative labels
    FullX = pd.read_csv(src_path,chunksize=ch_sz)
    for X in FullX:
        SS.transform(X[features])
        predictions = model.predict(X[features])
        Ts = np.array(X[labels]==1)
        Fs = np.logical_not(Ts)
        TNP += sum(Ts)
        TNN += sum(Fs)
        TP += np.sum(predictions[Ts])
        TN += np.sum(1-predictions[Fs])
    return (float(TP/TNP), float(TN/TNN))

def ModelCrossVal(model,src_path,ch_sz,features,labels,SS):  
    # Cross-Validation for parameter tuning
    kfold = 0 # number of folds (same as number of chunks)
    FullX = pd.read_csv(src_path,chunksize=ch_sz)
    for X in FullX:
        kfold+=1
            
    score = []
    for k in range(0,kfold):
        # Training
        FullX = pd.read_csv(src_path,chunksize=ch_sz)
        count = 0
        for X in FullX:
            if not count == k:
                SS.transform(X[features])
                model.partial_fit(X[features],X[labels],classes=classes)
                count+=1
        # Validation
        FullX = pd.read_csv(src_path,chunksize=ch_sz)       
        count = 0
        for X in FullX:
            if count == k:
                SS.transform(X[features])
                score[k] = model.score(X[features],X[labels])
                count+=1      
                
    return np.mean(score)
    
def ModelValFitScore(model,src_path,ch_sz,features,labels,SS): 
    # Model validation using the first chunk          
    # Training
    FullX = pd.read_csv(src_path,chunksize=ch_sz)
    count = 0
    for X in FullX:
        if not count == 0:
            SS.transform(X[features])
            model.partial_fit(X[features],X[labels],classes=classes)
            count+=1
    # Validation
    FullX = pd.read_csv(src_path,chunksize=ch_sz)       
    count = 0
    for X in FullX:
        if count == 0:
            SS.transform(X[features])            
            score = model.score(X[features],X[labels])
            count+=1      
                
    return score
    
    