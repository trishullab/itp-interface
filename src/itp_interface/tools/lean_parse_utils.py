#!/usr/bin/env python3

import typing
from itp_interface.tools.iter_helpers import ClonableIterator
from itp_interface.lean_server.lean_utils import Lean3Utils

class LeanLineByLineReader(object):
    class LineByLineIterator(ClonableIterator):
        def __init__(self, lines: typing.List[str]):
            self.lines = lines
            self.current_index = 0
        
        def __iter__(self) -> typing.Iterator[str]:
            return self
        
        def __next__(self) -> str:
            if self.current_index >= len(self.lines):
                raise StopIteration()
            line = self.lines[self.current_index]
            self.current_index += 1
            return line
        
        def set_to_index(self, index: int):
            assert 0 <= index < len(self.lines), f"Index {index} out of bounds for lines of length {len(self.lines)}"
            self.current_index = index
        
        def clone(self) -> 'LeanLineByLineReader.LineByLineIterator':
            cloned_iterator = LeanLineByLineReader.LineByLineIterator(self.lines)
            cloned_iterator.current_index = self.current_index
            return cloned_iterator

    def __init__(self, file_name: str = None, file_content: str = None, remove_comments: bool = False, no_strip: bool = False):
        assert file_name is not None or file_content is not None, "Either file_name or file_content must be provided"
        assert file_name is None or file_content is None, "Only one of file_name or file_content must be provided"
        self.file_name : str = file_name
        self.file_content : str = file_content
        self.no_strip = no_strip
        if self.file_name is not None:
            with open(file_name, 'r') as fd:
                self.file_content : str = fd.read()
        if remove_comments:
            self.file_content = Lean3Utils.remove_comments(self.file_content)

    def instruction_step_generator(self) -> ClonableIterator:
        lines = self.file_content.split('\n')
        if not self.no_strip:
            lines = [line.strip() for line in lines]
        return LeanLineByLineReader.LineByLineIterator(lines)