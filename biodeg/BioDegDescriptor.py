from . import BioDegClassifier
from mordred import Descriptor

class BioDegDescriptor(Descriptor):

    def __init__(self):
        c = BioDegClassifier.Prod()
        c.load()
        self.classifier = c
        
    def calculate(self):
        for k in dir(self.classifier):
            print(k)
        self.classifier.loadMols([self.mol]) 
        result = self.classifier.guess()
        return self.classifier.biodeg_string_from_state(result[0])
        
    def parameters(self):
        return ()