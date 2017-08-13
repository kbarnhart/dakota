"""An abstract base class for all Dakota variable types."""

from abc import ABCMeta, abstractmethod
from ..utils import to_iterable


class VariablesBase(object):

    """Describe features common to all Dakota variable types."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 variables='continuous_design',
                 descriptors=(),
                 scale_types = None,
                 scales = None,
                 **kwargs):
        """Create default variables parameters.

        Parameters
        ----------
        descriptors : str or tuple or list of str, optional
            Labels for the variables.
        variables : str, optional
            The type of parameter set (default is 'continuous_design').

        """
        self.variables = variables
        self._descriptors = descriptors
        self._scale_types = scale_types
        self._scales = scales

    @property
    def descriptors(self):
        """Labels attached to Dakota variables."""
        return self._descriptors
    
    @property
    def scale_types(self):
        """Scale types attached to Dakota variables."""
        return self._scale_types
    
    @property
    def scales(self):
        """Scales attached to Dakota variables."""
        return self._scales
    
    @descriptors.setter
    def descriptors(self, value):
        """Set labels for Dakota variables.

        Parameters
        ----------
        value : str or list or tuple of str
          The new variables labels.

        """
        if type(value) is str:
            value = (value,)
        if not isinstance(value, (tuple, list)):
            raise TypeError("Descriptors must be a string, tuple or list")
        self._descriptors = value

    @scale_types.setter
    def scale_types(self, value):
        """Set scale_types for Dakota variables.
            
            Parameters
            ----------
            value : str or list or tuple of str
            The new scale_types.
            
            """
        if type(value) is str:
            value = (value,)
        if not isinstance(value, (tuple, list)):
            raise TypeError("scale_types must be a string, tuple or list")
        self._scale_types = value
    
    @scales.setter
    def scales(self, value):
        """Set scale for Dakota variables.
            
            Parameters
            ----------
            value : str or list or tuple of str
            The new scale_types.
            
            """
        if type(value) is str:
            value = (value,)
        if not isinstance(value, (tuple, list)):
            raise TypeError("Scales must be a string, tuple or list")
        self._scales = value

    def __str__(self):
        """Define the variables block of a Dakota input file."""
        descriptors = to_iterable(self.descriptors)
        scale_types = to_iterable(self.scale_types)
        scales = to_iterable(self.scales)
        
        s = 'variables\n' \
            + '  {0} = {1}\n'.format(self.variables, len(descriptors))
        s += '    descriptors ='
        for vd in descriptors:
            s += ' {!r}'.format(vd)
            
        if self.scales is not None:
            s += '\n' \
                + '    scales ='
            for sc in scales:
                s += ' {!r}'.format(sc)
                
        if self.scale_types is not None:
            s += '\n' \
                + '    scale_type ='
            for sct in scale_types:
                s += ' {!r}'.format(sct)
        return(s)
