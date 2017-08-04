import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from DataModelsFunc import *

WD = os.getcwd()
train_OHC_path = os.path.join(WD,'train','train_OHC.csv')
train_binary_path = os.path.join(WD,'train','train_binary.csv')
test_OHC_path = os.path.join(WD,'test','test_OHC.csv')
test_binary_path = os.path.join(WD,'test','test_binary.csv')

ch_sz = 5000 # chunk size
epochs = 25 # number of epochs

# For this demo, I will only run the binary encoded files
src_path = train_binary_path
test_path = test_binary_path
# classifiers that allow for partial_fit in scikit-learn
models = {'MLP1' : MLPClassifier(hidden_layer_sizes=(200),alpha=0.1),
          'MLP2' : MLPClassifier(hidden_layer_sizes=(400),alpha=0.1),
          'MLP3' : MLPClassifier(hidden_layer_sizes=(200,200),alpha=0.1)}

FullX = pd.read_csv(src_path,chunksize=ch_sz)
X = FullX.get_chunk(1)
labels = ['income_level'] # income_level is the label of the dataset 
bin_features = X.columns.difference(labels)
labels = labels[0]
FullX.close()

# Normalizing/Scaling
SS = StandardScaler()
FullX = pd.read_csv(src_path,chunksize=ch_sz)
for X in FullX:
    SS.partial_fit(X[bin_features])

# Model Fitting - Testing
BaseScore, BaseTF = BaselineScore(test_path,ch_sz,labels)
scores = {'BaseLine' : BaseScore}
TF = {'BaseLine' : BaseTF} # (Correct Positives (1's) , Correct Negatives (0's))
epoch_scores = {} # variation of scores with each epoch
for mod in models:
    print('Training ' + mod + ' Classifier')
    epoch_scores[mod] = ModelFitOverSample(models[mod],src_path,ch_sz,epochs,bin_features,labels,SS)
    # You can also use:
    #    ModelFit or ModelFitUnderSample
    
    scores[mod] = ModelTestScore(models[mod],test_path,ch_sz,bin_features,labels,SS)
    TF[mod] = ModelTestFalseTrue(models[mod],test_path,ch_sz,bin_features,labels,SS)

PrintScores(scores,TF)
PlotEpochScores(epoch_scores,scores)