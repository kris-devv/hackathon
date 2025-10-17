class FixedSizeQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def push(self, item):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)  
        self.queue.append(item)

    def get_items(self):
        return list(self.queue)