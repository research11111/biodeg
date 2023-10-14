from . import BioDegClassifier
from mordred import Descriptor

class BioDegDescriptor(Descriptor):

    def __init__(self):
        c = BioDegClassifier.Prod()
        c.load()
        self.classifier = c
        
    def calculate(self):
        self.classifier.loadMols([self.mol]) 
        result = self.classifier.guess()
            
        return next(iter(result.values()))

        
    def parameters(self):
        return ()
        
    def description(self):
        return "Classify the molecule as being Ready Biodegradable (1) or Non Ready Biodegradable (0)"