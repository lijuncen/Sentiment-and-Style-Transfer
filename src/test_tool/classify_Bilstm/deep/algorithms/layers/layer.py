from abc import ABCMeta, abstractmethod

class layer:
    __metaclass__ = ABCMeta
    
    def _p(self, prefix, name):
        """
        Get the name with prefix.
        """
        return '%s_%s' % (prefix, name)
    
    @abstractmethod
    def getOutput(self, inputs):
        """
            :Return the output of the layer instance for a given input.
        """
