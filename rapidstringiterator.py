import comfy
import hashlib


# Alternative version that allows configuring wrap-around behavior
class RapidStringIterator:
    """
    Version that throws exception when reaching the end, no wrap-around
    """
    
    _node_states = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"default": "", "multiline": True}),
                "reset": ("BOOLEAN", {"default": False}),
                "node_id": ("STRING", {"default": "iterator_nowrap"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("current_string", "current_index", "is_last")
    FUNCTION = "iterate"
    CATEGORY = "custom/iterators"
    
    def iterate(self, string_list, reset=False, node_id="iterator_nowrap"):
        strings = [s.strip() for s in string_list.split('\n') if s.strip()]
        
        # Initialize state
        if node_id not in self._node_states or reset:
            self._node_states[node_id] = {
                'current_index': 0,
                'completed': False
            }
        
        state = self._node_states[node_id]
        
        if not strings:
            raise ValueError("String list is empty")
        
        # Check if iteration has completed
        if state['completed']:
            raise StopIteration(f"Iteration completed for node '{node_id}'. All {len(strings)} items have been returned. Use reset=True to restart.")
        
        # Get current string
        current_index = state['current_index']
        current_string = strings[current_index]
        is_last = (current_index == len(strings) - 1)
        
        # Update state
        if is_last:
            state['completed'] = True
        else:
            state['current_index'] += 1
        
        print(f"StringListIteratorNoWrap '{node_id}': Index {current_index}/{len(strings)}: '{current_string}'")
        
        return (current_string, current_index, is_last)