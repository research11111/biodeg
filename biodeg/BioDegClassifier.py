from rdkit import Chem
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from .BioDegClassifierModel import *
import csv

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    results = np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','S','F','Si','P','Cl','Br','Mg',
                                                               'Na','Ca','Fe','As','Al','I','B','V','K','Tl',
                                                               'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H',  # H?
                                                               'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
                                                               'Pt','Hg','Pb','Unknown']) + 
                       one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 
                       one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + 
                       [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + 
                       one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, 
                                                                       Chem.rdchem.HybridizationType.SP2, 
                                                                       Chem.rdchem.HybridizationType.SP3, 
                                                                       Chem.rdchem.HybridizationType.SP3D, 
                                                                       Chem.rdchem.HybridizationType.SP3D2]) + 
                       [atom.GetIsAromatic()])
    return np.array(results)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def mol2vec(mol):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features(atom) for atom in atoms]
    edge_index = get_bond_pair(mol)
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data
    
def df_check(df):
    df['mol2vec'] = 'Yes'
    for i in range(df.shape[0]):
        try:
            m = Chem.MolFromSmiles(df['SMILES'].iloc[i])
            vec = mol2vec(m)
    #         print(i, m, vec)
        except:
            df['mol2vec'].iloc[i] = 'No'
    #         print('No')
            continue

    df = df[df['mol2vec'] != 'No'].sample(frac=1).reset_index(drop=True)
    del df['mol2vec']
    return df
def make_mol(df):
    mols = {}
    for i in range(df.shape[0]):
        mols[Chem.MolFromSmiles(df['SMILES'].iloc[i])] = df['Class'].iloc[i]
    return mols
def make_vec(mols):
    X = [mol2vec(m) for m in mols.keys()]
    for i, data in enumerate(X):
        y = list(mols.values())[i]
        data.y = torch.tensor([y], dtype=torch.long)
    return X
def SMILES_to_InChIKey(df, col, i):
    try:
        s = df[col].iloc[i]
        m = Chem.MolFromSmiles(s)
        inc = Chem.inchi.MolToInchi(m)
        key = Chem.inchi.InchiToInchiKey(inc)
    except:
        key = np.nan
    return key
class BioDegClassifier:

    def __init__(self):
        paser = argparse.ArgumentParser()
        args = paser.parse_args("")
        args.optim = 'Adam'
        args.step_size = 10
        args.gamma = 0.9
        args.dropout = 0.1
        args.n_features = 70
        dim = 512
        args.conv_dim1 = dim
        args.conv_dim2 = dim
        args.conv_dim3 = dim
        args.concat_dim = dim
        args.pred_dim1 = dim
        args.pred_dim2 = dim
        args.out_dim = 2
        args.seed = 123
        self.args = args

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(args)
        self.model = model.to(self.device)

    def loadData(self,file):
        pass
    
    def load(self,file="data/models/model.pt"):
        model_state_dict = torch.load(file)
        self.model.load_state_dict(model_state_dict)
            
class Dev(BioDegClassifier):
    def __init__(self):
        super().__init__()
        self.test_data_proportion = 0.2
        self.random_state = 0
        self.train_data = None
        self.test_data = None

    def print(self,message,title="dev"):
        print(f'[{title}] {message}')

    def debug(self,message,title="dev debug"):
        self.print(message,title)
    
    def train(self, epoch=400, lr=0.0001):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(),lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.args.step_size,
                                              gamma=self.args.gamma)
        self.print('Training model')
        for epoch in range(epoch):
            scheduler.step()

            train_correct = 0
            train_total = 0
            epoch_train_loss = 0
            for i, data in enumerate(self.train_data):
                data = data.to(self.device)
                labels = data.y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                outputs.require_grad = False
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_train_loss /= len(self.train_data)
            train_acc =  100 * train_correct / train_total
            
            self.debug('epoch %d accuracy %.4f%% loss=%.4f' % (epoch+1,train_acc,epoch_train_loss))

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_data:
                data = data.to(self.device)
                labels = data.y.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_acc = 100 * correct / total
        return total_acc

    def loadData(self,file):
        df = pd.read_csv(file, low_memory=False)
        df = pd.concat([df['SMILES'], df['Class']], axis=1)

        X_train, X_test = train_test_split(df, test_size=self.test_data_proportion, shuffle=True, stratify=df['Class'], random_state=self.random_state)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        train_mols = make_mol(X_train)
        test_mols = make_mol(X_test)

        train_X = make_vec(train_mols)
        test_X = make_vec(test_mols)

        self.train_data = DataLoader(train_X, batch_size=len(train_X), shuffle=True, drop_last=True)
        self.test_data = DataLoader(test_X, batch_size=len(test_X), shuffle=False, drop_last=True)
    
    def save(self, file="data/models/model.pt"):
        torch.save(self.model.state_dict(), file)

class Prod(BioDegClassifier):
    def __init__(self):
        super().__init__()
        self.loadedData = None
        self.loadedMols = None

    # Load data from a .csv file and input into a dataloader
    # molecule to graph conversion is performed
    def loadData(self,file):
        df = pd.read_csv(file, low_memory=False)
        self.loadDataFrame(pd.concat([df['SMILES'], df['Class']], axis=1))
                
    def loadMols(self,mols):
        df = pd.DataFrame({'SMILES': [Chem.MolToSmiles(mol) for mol in mols], 'Class': [0]})
        self.loadDataFrame(df)
        
    def loadDataFrame(self,df):
        mols = make_mol(df)
        vectors = make_vec(mols)
        self.loadedMols = list(mols.keys())
        self.loadedData = DataLoader(vectors, batch_size=len(vectors), shuffle=False, drop_last=False)

    def unloadData(self):
        self.loadedData = None
        self.loadedMols = None

    def guess(self):
        self.model.eval()
        pred_data_total = []

        with torch.no_grad():
            for data in self.loadedData:
                data = data.to(self.device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                pred_data_total += predicted.view(-1).cpu().tolist()
                
        result = {}
        for idx in range(len(pred_data_total)):
            associatedMol = self.loadedMols[idx]
            result[associatedMol] = pred_data_total[idx]
    
        return result
    
    def biodeg_string_from_state(self,state):
        return 'RB' if state == 1 else 'NRB'
        
    def guess_pretty_print(self,result):
        for key in result.keys():
            biodeg = self.biodeg_string_from_state(result[key])
            print('%s : %s' % (Chem.MolToSmiles(key), biodeg))
    
    def guess_result_to_csv(self,file,result):
        with open(file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["SMILES", "BioDegradability"])
            for key, value in result.items():
                writer.writerow([Chem.MolToSmiles(key), value])