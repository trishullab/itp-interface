"""
Ultra-simple tactic parser for Lean 4 code.

This module provides a lightweight interface to parse tactics from Lean 4 proofs
without compiling or running any code - just pure syntax parsing.

The parser process runs in the background to avoid restart overhead.
"""

import base64
import json
import os
import subprocess
import logging
from bisect import bisect_left
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import List, Dict, Optional

class Position(BaseModel):
    """Represents a position in the source code."""
    line: int
    column: int

    @field_validator('line', 'column')
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Line and column numbers must be non-negative")
        return v
    
    def __lt__(self, other: 'Position') -> bool:
        if self.line == other.line:
            return self.column < other.column
        return self.line < other.line
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Position):
            return NotImplemented
        return self.line == value.line and self.column == value.column
    
    def __le__(self, other: 'Position') -> bool:
        return self < other or self == other

    def is_contained_in(self, start: 'Position', end: 'Position') -> bool:
        """Check if this position is within the range [start, end]."""
        return start <= self <= end

class TreeNode(BaseModel):
    """Represents a node in the syntax tree."""
    type: str
    text: Optional[str] = None
    start_pos: Optional[Position] = None
    end_pos: Optional[Position] = None
    children: List['TreeNode'] = []

    # Make sure to test the `type` field properly
    @field_validator('type')
    def validate_type(cls, v):
        if not isinstance(v, str) or not v:
            raise ValueError("Type must be a non-empty string")
        # the type must be `context`, `tacticInfo`, `other`, or `hole`
        if v not in {'context', 'tacticInfo', 'other', 'hole'}:
            raise ValueError(f"Invalid type: {v}")
        return v
    
    def __lt__(self, other: 'TreeNode') -> bool:
        if self.start_pos is None or other.start_pos is None:
            return False
        if self.start_pos == other.start_pos:
            if self.end_pos is None or other.end_pos is None:
                return False
            return self.end_pos < other.end_pos
        return self.start_pos < other.start_pos
    
    def __le__(self, other: 'TreeNode') -> bool:
        return self < other or self == other
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TreeNode):
            return NotImplemented
        return (self.type == value.type and
                self.start_pos == value.start_pos and
                self.end_pos == value.end_pos)
    
    
    def is_contained_in(self, tree_node: 'TreeNode') -> bool:
        """Check if this node is contained within another node's position range."""
        if self.start_pos is None or self.end_pos is None:
            return False
        if tree_node.start_pos is None or tree_node.end_pos is None:
            return False
        return (self.start_pos.is_contained_in(tree_node.start_pos, tree_node.end_pos) and
                self.end_pos.is_contained_in(tree_node.start_pos, tree_node.end_pos))

class TacticInfo:
    """Information about a single tactic."""

    def __init__(self, text: str, line: int, column: int, end_line: int, end_column: int):
        self.text = text
        self.line = line
        self.column = column
        self.end_line = end_line
        self.end_column = end_column

    def __repr__(self) -> str:
        return f"TacticInfo(text={self.text!r}, line={self.line}, column={self.column})"

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "line": self.line,
            "column": self.column,
            "endLine": self.end_line,
            "endColumn": self.end_column
        }


class TacticParser:
    """Parse tactics from Lean 4 code without compilation.

    The parser process runs in the background and is reused across multiple requests.

    If you want to parse tactics that use mathlib or other dependencies, provide a
    project_path when initializing the parser. The process will run from that directory
    and automatically find the project's .lake/build with all dependencies.
    """

    def __init__(self, parser_path: Optional[str] = None, project_path: Optional[str] = None, file_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the tactic parser.

        Args:
            parser_path: Path to the tactic-parser executable. If None, uses the default path.
            project_path: Path to a Lean project directory (contains lakefile.toml and .lake/build).
                         If provided, the parser will run from this directory and can use the
                         project's dependencies (like mathlib). If None, uses minimal environment.
            logger: Optional logger for debugging
        """
        if parser_path is None:
            # Default path relative to this file
            default_path = Path(__file__).parent / "tactic_parser" / ".lake" / "build" / "bin" / "tactic-parser"
            self.parser_path = str(default_path)
        else:
            self.parser_path = parser_path

        self.project_path = project_path
        self.file_path = file_path
        self.logger = logger if logger else logging.getLogger(__name__)
        self.process: Optional[subprocess.Popen] = None
        self._start()

    def _start(self):
        """Start the tactic parser process."""
        try:
            # Determine working directory:
            # - If project_path provided: use project directory (finds .lake/build automatically)
            # - Otherwise: use tactic_parser directory (minimal environment)
            if self.project_path:
                working_dir = self.project_path
                self.logger.debug(f"Starting parser in project mode from: {working_dir}")
            else:
                working_dir = Path(self.parser_path).parent.parent.parent
                self.logger.debug(f"Starting parser in standalone mode from: {working_dir}")
            tools_dir = os.path.dirname(__file__)
            repl_path = os.path.join(tools_dir, "tactic_parser")
            abs_path = os.path.abspath(repl_path)
            path_to_repl_exec = os.path.join(abs_path, ".lake", "build", "bin", "tactic-parser")
            cmds = ["lake", "env", path_to_repl_exec]
            if self.file_path:
                cmds += [self.file_path]
            self.process = subprocess.Popen(
                cmds,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=str(working_dir)
            )
            self.logger.debug(f"Started tactic parser process (PID: {self.process.pid})")
        except FileNotFoundError:
            raise RuntimeError(
                f"Tactic parser not found at {self.parser_path}. "
                f"Please build it first with: cd {Path(self.parser_path).parent.parent.parent} && lake build"
            )

    def _ensure_running(self):
        """Ensure the process is running, restart if needed."""
        if self.process is None or self.process.poll() is not None:
            self.logger.warning("Tactic parser process died, restarting...")
            self._start()

    def _add_overlapping_tactics(self, tactic_context: List[TreeNode], tree: TreeNode):
        assert tree.type == "tacticInfo"
        assert tree.start_pos is not None
        assert tree.end_pos is not None
        # Find the tactic context that overlaps with the given position
        idx = bisect_left(tactic_context, tree)
        if len(tactic_context) == idx:
            tactic_context.append(tree)
            return

        tree_idx = tactic_context[idx]
        if tree.is_contained_in(tree_idx):
            # Replace the tactic all together
            tactic_context[idx] = tree
        else:
            tactic_context.insert(idx, tree)




    def _collect_tactics_from_tree(self, tree: TreeNode, tactic_context: list = []) -> List[TreeNode]:
        """Recursively extract TacticInfo from the syntax tree."""
        if tree.type == "tacticInfo" and tree.text and tree.start_pos and tree.end_pos:
            self._add_overlapping_tactics(tactic_context, tree)

        for child in tree.children:
            self._collect_tactics_from_tree(child, tactic_context)

        return tactic_context


    def parse(self, lean_code: str) -> List[TacticInfo]:
        """
        Parse tactics from Lean 4 code.

        Args:
            lean_code: Lean 4 source code as a string

        Returns:
            List of TacticInfo objects

        Raises:
            RuntimeError: If parsing fails
        """
        self._ensure_running()

        retry_cnt = 5
        succeeded = False

        while retry_cnt > 0 and not succeeded:
            # Encode Lean code as base64
            b64_input = base64.b64encode(lean_code.encode('utf-8')).decode('ascii')
            self.logger.debug(f"Sending {len(lean_code)} bytes of Lean code")

            # Send to parser
            try:
                self.process.stdin.write(b64_input + '\n')
                self.process.stdin.flush()
                succeeded = True
            except BrokenPipeError:
                self.logger.error("Broken pipe, restarting process")
                self._start()
            retry_cnt -= 1

        # Read JSON response (one line)
        try:
            response_line = self.process.stdout.readline()
            self.logger.debug(f"Response: {response_line.strip()}")
            if not response_line:
                # Check stderr for error messages
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"Parser process died unexpectedly. Stderr: {stderr_output}")
        except Exception as e:
            raise RuntimeError(f"Failed to read response: {e}")

        # Parse JSON response
        try:
            response = json.loads(response_line.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON response: {e}\nOutput: {response_line}")

        # Check for errors
        if response.get("error"):
            raise RuntimeError(f"Parse error: {response['error']}")

        # Convert tree to TacticInfo objects
        trees = []
        tactics = []
        for t in response.get("trees", []):
            if t is not None:
                tree = TreeNode.model_validate(t)
                self._collect_tactics_from_tree(tree, trees)
        for t in trees:
            assert t.start_pos is not None
            assert t.end_pos is not None
            tactics.append(
                TacticInfo(
                    text=t.text if t.text else "",
                    line=t.start_pos.line,
                    column=t.start_pos.column,
                    end_line=t.end_pos.line,
                    end_column=t.end_pos.column
                )
            )
        self.logger.debug(f"Parsed {len(tactics)} tactics")
        # Remove the `by` tactic if present at the start
        if len(tactics) > 0 and tactics[0].text.strip() == "by":
            tactics = tactics[1:]
            self.logger.debug("Removed leading 'by' tactic")
        return tactics

    def parse_file(self, file_path: str) -> List[TacticInfo]:
        """
        Parse tactics from a Lean 4 file.

        Args:
            file_path: Path to the Lean 4 file

        Returns:
            List of TacticInfo objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lean_code = f.read()
        return self.parse(lean_code)

    def close(self):
        """Close the parser process."""
        if self.process:
            try:
                # Send exit command
                self.process.stdin.write('\n')
                self.process.stdin.flush()
                self.process.wait(timeout=1)
            except:
                self.process.kill()
            self.process = None
            self.logger.debug("Closed tactic parser process")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# Example usage
def print_tactics(tactics: List[TacticInfo]):
    for tactic in tactics:
        print(f"Line {tactic.line}, Col {tactic.column} to Line {tactic.end_line}, Col {tactic.end_column}: {tactic.text}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    with TacticParser() as parser:
        # Example 1: Simple proof
        lean_code = "example : True := by trivial"

        print("Parsing example 1...")
        tactics = parser.parse(lean_code)
        print_tactics(tactics)


    project_path = str(Path(__file__).parent.parent.parent / "data" / "test" / "lean4_proj")
    file_path = str(Path(__file__).parent.parent.parent / "data" / "test" / "lean4_proj" / "Lean4Proj"/ "Basic.lean")
    file_path = None

    with TacticParser(project_path=project_path, file_path=file_path) as parser:
        # Example 2: Multiline with params
        lean_code2 = "example (r: Nat) (p q : Prop) (hp : p) (hq : q) : p ∧ q := by\n  apply And.intro\n  exact hp\n  exact hq"

        print("\nParsing example 2...")
        tactics2 = parser.parse(lean_code2)
        print_tactics(tactics2)
        
        # Check if linarith is parsed correctly
        lean_code3 = """
import Mathlib
        
example (a b : Nat) 
(h1: a + b = 10)
(h2: a = 5) :
b = 5:= by
  rw [h2] at h1
  linarith
"""
        print("\nParsing example 3...")
        tactics3 = parser.parse(lean_code3)
        print_tactics(tactics3)

