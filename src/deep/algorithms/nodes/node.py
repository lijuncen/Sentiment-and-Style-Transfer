from abc import ABCMeta, abstractmethod

class Node:
    __metaclass__ = ABCMeta
    
    def _p(self, prefix, name):
        """
        Get the name with prefix.
        """
        return '%s_%s' % (prefix, name)
    
    @abstractmethod
    def node_update(self, input):
        """
        Manipulate in the node.
        """