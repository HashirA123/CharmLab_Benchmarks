
from typing import Any, Tuple, List, Union, Optional

#NOTE not being used right now

class DataAttributes:
    """
    This class is responsible for loading and storing the dataset attributes based on the provided configuration file.
    It serves as a centralized place to access feature metadata, which can be used by other modules for processing, modeling, and explanation.
    """
    def __init__(self,
                short_name: str,
                type: str,
                node_type: str,
                actionability: str,
                mutability: bool,
                encode: Optional[str] = None,
                impute: Optional[str] = None,
                domain: Optional[Union[List[Any], Tuple[Any, Any]]] = None,):
        self.short_name = short_name
        self.type = type
        self.node_type = node_type
        self.actionability = actionability
        self.mutability = mutability
        self.encode = encode
        self.impute = impute
        self.domain = domain