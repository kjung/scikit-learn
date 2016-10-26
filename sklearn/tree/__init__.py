"""
The :mod:`sklearn.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import DecisionTreeClassifier
from .tree import DecisionTreeRegressor
from .tree import ExtraTreeClassifier
from .tree import ExtraTreeRegressor
from .tree import PropensityTree
from .tree import PowersTree
from .tree import DoubleSampleTree
from .export import export_graphviz

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor",
           "ExtraTreeClassifier", "ExtraTreeRegressor",
           "PropensityTree", "PowersTree", "DoubleSampleTree",
           "export_graphviz"]
