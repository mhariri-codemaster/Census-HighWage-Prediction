# I modified thios from the original version which used sparse representation
import numpy as np

class OneHotEncoder():
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a binary matrix doing one-of-k encoding.

    Parts of code borrowed from Paul Duan (www.paulduan.com)
    Licence: MIT (https://github.com/pyduan/amazonaccess/blob/master/MIT-LICENSE)
    
    Source:https://kaggle2.blob.core.windows.net/forum-message-attachments/
    51794/1444/OneHotEncoder.py?sv=2015-12-11&sr=b&sig=AriapnTxan%2B4lKUYh7sWc
    c22G%2FqLTOFF0DvhoslZKbo%3D&se=2017-07-24T00%3A36%3A48Z&sp=r
    
    variables: 
        keymap: list of dict
        cat_col: list
        num_col: list
        new_col: list
    """

    def __init__(self):
        self.keymap = None
        self.cat_col = None
        self.num_col = None
        self.new_col = None

    def fit(self, x):
        self.keymap = []
        for col in x.columns:
            uniques = set(list(x[col]))
            self.keymap.append(dict((key, i) for i, key in enumerate(uniques)))

    def partial_fit(self, x):
        """
        This method can be used for doing one hot encoding in mini-batch mode.
        """
        if self.keymap is None:
            self.fit(x)
        else:
            for i, col in enumerate(x.columns):
                uniques = set(list(self.keymap[i].keys()) + (list(x[col])))
                self.keymap[i] = dict((key, i) for i, key in enumerate(uniques))

    def transform(self, x):
        if self.keymap is None:
            self.fit(x)

        outdat = []
        for i, col in enumerate(x.columns):
            km = self.keymap[i]
            num_labels = len(km)
            mat = np.zeros((x.shape[0], num_labels))
            for j, val in enumerate(x[col]):
                if val in km:
                    mat[j, km[val]] = 1
            outdat.append(mat)
        outdat = np.concatenate(outdat, axis=1)
        return outdat
        
    def set_columns(self, cat_col,num_col,new_col):
        self.cat_col = cat_col
        self.num_col = num_col
        self.new_col = new_col
        
class BinaryEncoder():
    """
    BinaryEncoder takes data matrix with categorical columns and
    converts it to a binary matrix using binary encoding. I wrote this 
    one myself
    
    variables: 
        keymap: list of dict
        cat_col: list
        num_col: list
        new_col: list
    """

    def __init__(self):
        self.keymap = None
        self.cat_col = None
        self.num_col = None
        self.new_col = None

    def fit(self, x):
        self.keymap = []
        for col in x.columns:
            uniques = set(list(x[col]))
            self.keymap.append(dict((key, i) for i, key in enumerate(uniques)))

    def partial_fit(self, x):
        """
        This method can be used for doing binary encoding in mini-batch mode.
        """
        if self.keymap is None:
            self.fit(x)
        else:
            for i, col in enumerate(x.columns):
                uniques = set(list(self.keymap[i].keys()) + (list(x[col])))
                self.keymap[i] = dict((key, i) for i, key in enumerate(uniques))

    def transform(self, x):
        if self.keymap is None:
            self.fit(x)

        outdat = []
        for i, col in enumerate(x.columns):
            km = self.keymap[i]
            num_labels = len(bin(len(km)))-2
            mat = np.zeros((x.shape[0], num_labels))
            for j, val in enumerate(x[col]):
                if val in km:
                    binval = list(bin(km[val])[2:])
                    for k in range(len(binval)):
                        mat[j,-k-1] = int(binval[-k-1])
            outdat.append(mat)
        outdat = np.concatenate(outdat, axis=1)
        return outdat
        
    def set_columns(self, cat_col,num_col,new_col):
        self.cat_col = cat_col
        self.num_col = num_col
        self.new_col = new_col