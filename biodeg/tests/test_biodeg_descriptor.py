import pytest
from rdkit import Chem
from mordred import Calculator, descriptors

from biodeg.BioDegDescriptor import *

def smiles_expect(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    
    assert mol is not None
    
    d = BioDegDescriptor()
    calc = Calculator(descs=[d])
    result = calc(mol)
    
    assert result[d] == expected

def test_calculate_biodeg_descriptor_1():
    smiles_expect('c1ccc2cc3ccccc3cc2c1',0)
def test_calculate_biodeg_descriptor_2():
    smiles_expect('c1ccccc1',1)
def test_calculate_biodeg_descriptor_3():
    smiles_expect('c1ccncc1',1)
