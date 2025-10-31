import typing
from abc import ABC, abstractmethod

class ClonableIterator(ABC):
    @abstractmethod
    def __iter__(self) -> typing.Iterator[str]:
        pass

    @abstractmethod
    def __next__(self) -> str:
        pass

    @abstractmethod
    def set_to_index(self, index: int):
        pass

    @abstractmethod
    def clone(self) -> 'ClonableIterator':
        pass


class IntertwinedIterator(ClonableIterator):
    def __init__(self, iterator: typing.Optional[ClonableIterator] = None):
        self.base_iterator = iterator
        self.next_instruction: typing.Optional[str] = None
        self.base_iterator_stopped = iterator is None # if the base iterator is None, then it is stopped
    
    def set_next_instruction(self, instruction: str):
        assert self.next_instruction is None, "next_instruction must be None"
        assert instruction is not None, "instruction must not be None"
        self.next_instruction = instruction
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.next_instruction is not None:
            # Return the next instruction if it is set
            next_instruction = self.next_instruction
            self.next_instruction = None
            return next_instruction
        # Otherwise, get the next instruction from the base iterator
        if self.base_iterator is not None and not self.base_iterator_stopped:
            try:
                instruction = next(self.base_iterator)
                return instruction
            except StopIteration:
                self.base_iterator_stopped = True
                raise
        else:
            raise StopIteration()
    
    def set_to_index(self, index: int):
        if self.base_iterator is not None:
            self.base_iterator.set_to_index(index)
            self.base_iterator_stopped = False

    def clone(self) -> 'IntertwinedIterator':
        cloned_iterator = IntertwinedIterator()
        if self.base_iterator is not None:
            cloned_iterator.base_iterator = self.base_iterator.clone()
            cloned_iterator.base_iterator_stopped = self.base_iterator_stopped
        cloned_iterator.next_instruction = self.next_instruction
        return cloned_iterator
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass