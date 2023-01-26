from zope.interface import Interface


class AIAlgorithmInterface(Interface):
    def train(self):
        pass

    def extractFromModel(self, vector):
        pass
