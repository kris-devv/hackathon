from collections import deque

class FixedSizeQueue:
    """
    A fixed-size queue that automatically removes the oldest item when full.
    Implemented using collections.deque for efficient operations.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize a fixed-size queue.
        
        Args:
            max_size (int): Maximum number of items the queue can hold
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
    
    def push(self, item):
        """
        Add an item to the queue. If full, removes the oldest item.
        
        Args:
            item: Item to add to the queue
        """
        self.queue.append(item)
    
    def get_items(self) -> list:
        """
        Get all items in the queue as a list.
        
        Returns:
            list: All items in the queue, oldest to newest
        """
        return list(self.queue)
    
    def size(self) -> int:
        """
        Get the current number of items in the queue.
        
        Returns:
            int: Number of items currently in the queue
        """
        return len(self.queue)
    
    def is_full(self) -> bool:
        """
        Check if the queue is full.
        
        Returns:
            bool: True if queue is at maximum capacity
        """
        return len(self.queue) == self.max_size
    
    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        
        Returns:
            bool: True if queue has no items
        """
        return len(self.queue) == 0
    
    def clear(self):
        """
        Remove all items from the queue.
        """
        self.queue.clear()
    
    # NEW FUNCTIONS FROM fixed_sized_queue_kopia.py
    
    def peek(self):
        """
        Get the oldest item without removing it.
        
        Returns:
            Item at the front of the queue, or None if empty
        """
        if self.is_empty():
            return None
        return self.queue[0]
    
    def peek_last(self):
        """
        Get the newest item without removing it.
        
        Returns:
            Item at the back of the queue, or None if empty
        """
        if self.is_empty():
            return None
        return self.queue[-1]
    
    def get_range(self, start: int, end: int) -> list:
        """
        Get a slice of items from the queue.
        
        Args:
            start (int): Start index (inclusive)
            end (int): End index (exclusive)
            
        Returns:
            list: Items in the specified range
        """
        if start < 0 or end > len(self.queue) or start >= end:
            return []
        return list(self.queue)[start:end]
    
    def get_last_n(self, n: int) -> list:
        """
        Get the last N items from the queue.
        
        Args:
            n (int): Number of items to retrieve
            
        Returns:
            list: Last N items (or all items if N > size)
        """
        if n <= 0:
            return []
        n = min(n, len(self.queue))
        return list(self.queue)[-n:]
    
    def contains(self, item) -> bool:
        """
        Check if an item exists in the queue.
        
        Args:
            item: Item to search for
            
        Returns:
            bool: True if item is in the queue
        """
        return item in self.queue
    
    def to_list(self) -> list:
        """
        Convert queue to list (alias for get_items).
        
        Returns:
            list: All items in the queue
        """
        return self.get_items()
    
    def __repr__(self) -> str:
        """
        String representation of the queue.
        
        Returns:
            str: String showing queue contents and max size
        """
        return f"FixedSizeQueue(max_size={self.max_size}, items={list(self.queue)})"
    
    def __len__(self) -> int:
        """
        Get the length of the queue.
        
        Returns:
            int: Number of items in the queue
        """
        return len(self.queue)
    
    def __iter__(self):
        """
        Make the queue iterable.
        
        Returns:
            Iterator over queue items
        """
        return iter(self.queue)