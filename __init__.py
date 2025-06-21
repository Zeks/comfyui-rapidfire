import os

from .datastasher import CsvWriterNode
from .dataloader import ImmatureImageDataLoader
from .imagecounter import ImmatureImageCounter
from .cached_checkpoint import CachedCheckpoint
from .escapedfind import BracketEscaper
from .girlselector import RandomCharacterSelector
from .schedulerselector import *
from .advancedksampler import *
from .danbooru import Ranbooru


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "CsvWriterNode": CsvWriterNode,
    "ImmatureImageDataLoader": ImmatureImageDataLoader,
    "ImmatureImageCounter": ImmatureImageCounter,
    "CachedCheckpoint": CachedCheckpoint,
    "BracketEscaper": BracketEscaper,
    "RandomCharacterSelector": RandomCharacterSelector,
    "RapidSchedulerSelector": RapidSchedulerSelector,
    "RapidSchedulerCombo": RapidSchedulerCombo,
    "Ranbooru": Ranbooru,
    "MultiModelAdvancedKsampler": MultiModelAdvancedKsampler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CsvWriterNode": "CSV Writer",
    "ImmatureImageDataLoader": "Immature Image Data Loader",
    "ImmatureImageCounter": "Immature Image Counter",
    "CachedCheckpoint": "Cached Checkpoint",
    "BracketEscaper": "Bracket Escaper",
    "RapidSchedulerSelector": "Scheduler Selector",
    "RapidSchedulerCombo": "Rapid Scheduler Combo",
    "Ranbooru": "Danbooru",
    "MultiModelAdvancedKsampler": "MultiModelAdvancedKsampler"
}

__all__ = NODE_CLASS_MAPPINGS


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


