import pytest
from rdkit import Chem
from mordred import Calculator, descriptors

from biodeg.BioDegDescriptor import *

def test_calculate_biodeg_descriptor():
    mol = Chem.MolFromSmiles('CCO')
    
    assert mol is not None
    
    d = BioDegDescriptor()
    calc = Calculator(descs=[d])
    
    result = calc(mol)
    print(result)
    
    expected_result = 'EXPECTED_RESULT'
    assert result[BioDegDescriptor] == expected_result
