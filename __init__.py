import os

from .datastasher import CsvWriterNode
from .stringhasher import StringHasher
from .dataloader import ImmatureImageDataLoader
from .imagecounter import ImmatureImageCounter
from .cached_checkpoint import CachedCheckpoint
from .escapedfind import BracketEscaper
from .girlselector import RandomCharacterSelector
from .schedulerselector import *
from .advancedksampler import *
from .multimodelcheckpointiterator import *
from .multimodelcheckpointiteratorfirst import MultiModelCheckpointIteratorFirst

from .danbooru import Ranbooru


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "CsvWriterNode": CsvWriterNode,
    "StringHasher": StringHasher,
    "ImmatureImageDataLoader": ImmatureImageDataLoader,
    "ImmatureImageCounter": ImmatureImageCounter,
    "CachedCheckpoint": CachedCheckpoint,
    "BracketEscaper": BracketEscaper,
    "RandomCharacterSelector": RandomCharacterSelector,
    "RapidSchedulerSelector": RapidSchedulerSelector,
    "RapidSchedulerCombo": RapidSchedulerCombo,
    "Ranbooru": Ranbooru,
    "MultiModelAdvancedKsampler": MultiModelAdvancedKsampler,
    "MultiModelCheckpointIterator": MultiModelCheckpointIterator,
    "MultiModelPromptSaver": MultiModelPromptSaver,
    "MultiModelPromptSaverIterative": MultiModelPromptSaverIterative,
    "MultiModelPromptSaverIterativeFirst": MultiModelCheckpointIteratorFirst,
    "MultiModelPromptSaverIterativeFirst": MultiModelCheckpointIteratorFirst,
    
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CsvWriterNode": "CSV Writer",
    "StringHasher": "StringHasher",
    "ImmatureImageDataLoader": "Immature Image Data Loader",
    "ImmatureImageCounter": "Immature Image Counter",
    "CachedCheckpoint": "Cached Checkpoint",
    "BracketEscaper": "Bracket Escaper",
    "RapidSchedulerSelector": "Scheduler Selector",
    "RapidSchedulerCombo": "Rapid Scheduler Combo",
    "Ranbooru": "Danbooru",
    "MultiModelAdvancedKsampler": "MultiModelAdvancedKsampler",
    "MultiModelCheckpointIterator": "MultiModelCheckpointIterator",
    "MultiModelPromptSaver": "MultiModelPromptSaver",
    "MultiModelPromptSaverIterative": "MultiModelPromptSaverIterative",
    "MultiModelPromptSaverIterativeFirst": "MultiModelPromptSaverIterativeFirst"
}

__all__ = NODE_CLASS_MAPPINGS


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



