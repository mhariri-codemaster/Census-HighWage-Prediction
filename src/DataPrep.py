import os
from DataPrepFunc import *
import pickle
    
WD = os.getcwd()
train_path = os.path.join(WD,'train','train.csv') 
train_temp_path = os.path.join(WD,'train','temp.csv')
train_mod_path = os.path.join(WD,'train','train_modified.csv')
train_OHC_path = os.path.join(WD,'train','train_OHC.csv')
train_binary_path = os.path.join(WD,'train','train_binary.csv')
test_path = os.path.join(WD,'test','test.csv') 
test_temp_path = os.path.join(WD,'test','temp.csv')
test_mod_path = os.path.join(WD,'test','test_modified.csv')
test_OHC_path = os.path.join(WD,'test','test_OHC.csv')
test_binary_path = os.path.join(WD,'test','test_binary.csv')

ch_sz = 5000
FixData1(train_path,train_temp_path,ch_sz)
Tot_avg_wage = FindAvgWage(train_temp_path,ch_sz)   
FixData2(train_temp_path,train_mod_path,ch_sz,Tot_avg_wage)
os.remove(train_temp_path)
Enc_OHC = EncodeTrainData_OHC(train_mod_path,train_OHC_path,ch_sz)
Enc_bin = EncodeTrainData_binary(train_mod_path,train_binary_path,ch_sz)
with open('Encoders.pickle','wb') as f:
    pickle.dump([Enc_OHC,Enc_bin], f)
    
FixData1(test_path,test_temp_path,ch_sz)
FixData2(test_temp_path,test_mod_path,ch_sz,Tot_avg_wage)
os.remove(test_temp_path)
EncodeTestData(test_mod_path,test_OHC_path,ch_sz,Enc_OHC)
EncodeTestData(test_mod_path,test_binary_path,ch_sz,Enc_bin)

# To get the column names and map use
#with open('Encoders.pickle','rb') as f:
#    Enc = pickle.load(f)