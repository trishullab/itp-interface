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
import re
import shutil
from enum import Enum
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import List, Dict, Optional, Union

class Position(BaseModel):
    """Represents a position in the source code."""
    line: int # Line counting starts from 1
    column: int # Column counting starts from 0

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
    decl_type: Optional[str] = None
    name: Optional[str] = None
    doc_string: Optional[str] = None
    start_pos: Optional[Position] = None
    end_pos: Optional[Position] = None
    text: Optional[str] = None
    namespace: Optional[str] = None
    children: List['TreeNode'] = []
    
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
        return (self.start_pos == value.start_pos and
                self.end_pos == value.end_pos)
    
    
    def is_contained_in(self, tree_node: 'TreeNode') -> bool:
        """Check if this node is contained within another node's position range."""
        if self.start_pos is None or self.end_pos is None:
            return False
        if tree_node.start_pos is None or tree_node.end_pos is None:
            return False
        return (self.start_pos.is_contained_in(tree_node.start_pos, tree_node.end_pos) and
                self.end_pos.is_contained_in(tree_node.start_pos, tree_node.end_pos))

class ErrorInfo(BaseModel):
    """Represents an error in parsing."""
    message: str
    position: Position

    def to_json(self) -> str:
        # Use pydantic's built-in json method
        return ErrorInfo.model_dump_json(self)

class LeanLineInfo(BaseModel):
    """Information about a single tactic."""
    text: str
    line: int
    column: int
    end_line: int
    end_column: int
    decl_type: Optional[str] = None
    name: Optional[str] = None
    doc_string: Optional[str] = None
    namespace: Optional[str] = None

    def __repr__(self) -> str:
        return f"LeanLineInfo(text={self.text!r}, line={self.line}, column={self.column})"

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "line": self.line,
            "column": self.column,
            "endLine": self.end_line,
            "endColumn": self.end_column,
            "declType": self.decl_type,
            "name": self.name,
            "docString": self.doc_string,
            "namespace": self.namespace
        }


    def to_json(self, indent=0) -> str:
        if indent == 0:
            return self.model_dump_json()
        else:
            return self.model_dump_json(indent=indent)

    @staticmethod
    def load_from_file(file_path: str):
        raise NotImplementedError("load_from_file must be implemented by the child class")
    
    @staticmethod
    def load_from_string(json_text: str):
        raise NotImplementedError("load_from_string must be implemented by the child class")

class DeclarationDependency(BaseModel):
    """Information about a single dependency reference."""
    name: str  # Fully qualified name (e.g., "Nat.add_zero")
    namespace: Optional[str] = None  # Namespace portion (e.g., "Nat")
    local_name: str  # Local name without namespace (e.g., "add_zero")
    file_path: Optional[str] = None  # Source file if resolvable
    module_name: Optional[str] = None  # Module where defined
    decl_id: Optional[str] = None  # Optional declaration ID for linking

    def __repr__(self) -> str:
        module_info = f" (from {self.module_name})" if self.module_name else ""
        return f"{self.name}{module_info}"

class DeclWithDependencies(BaseModel):
    """Declaration with its dependencies - designed for merging into larger collections."""
    decl_info: LeanLineInfo  # Declaration metadata
    dependencies: List[DeclarationDependency]
    unresolved_names: List[str] = []  # Names we couldn't resolve
    decl_id: Optional[str] = None  # Optional unique ID for this declaration

    def __repr__(self) -> str:
        id_info = f" [ID: {self.decl_id}]" if self.decl_id else ""
        return f"[{self.decl_info.decl_type}] {self.decl_info.name} ({len(self.dependencies)} deps){id_info}"

    def set_decl_id(self, decl_id: str) -> 'DeclWithDependencies':
        """Set declaration ID (useful for chaining)."""
        self.decl_id = decl_id
        return self
    
    def to_json(self, indent=0) -> str:
        if indent == 0:
            return self.model_dump_json()
        else:
            return self.model_dump_json(indent=indent)

    @staticmethod
    def from_dict(decl_data: Dict) -> 'DeclWithDependencies':
        """Create from dictionary (e.g., from JSON dependency parser output)."""
        decl_dict = decl_data['declaration']

        # Create LeanLineInfo from declaration data
        declaration = LeanLineInfo(
            text=decl_dict.get('text', ''),
            line=decl_dict.get('startPos', 0),
            column=0,
            end_line=decl_dict.get('endPos', 0),
            end_column=0,
            decl_type=decl_dict.get('declType'),
            name=decl_dict.get('name'),
            doc_string=decl_dict.get('docString'),
            namespace=decl_dict.get('namespace')
        )

        # Parse dependencies
        dependencies = []
        for dep_data in decl_data.get('dependencies', []):
            dependencies.append(DeclarationDependency(
                name=dep_data['name'],
                namespace=dep_data.get('namespace'),
                local_name=dep_data['localName'],
                file_path=dep_data.get('filePath'),
                module_name=dep_data.get('moduleName'),
                decl_id=dep_data.get('declId')
            ))

        return DeclWithDependencies(
            decl_info=declaration,
            dependencies=dependencies,
            unresolved_names=decl_data.get('unresolvedNames', []),
            decl_id=decl_data.get('declId')
        )

    @staticmethod
    def from_dependency_analysis(analysis_dict: Dict) -> List['DeclWithDependencies']:
        """
        Extract list of declarations from dependency parser output.

        Args:
            analysis_dict: Dict returned by parse_file with PARSE_DEPENDS

        Returns:
            List of DeclWithDependencies that can be merged into larger collections
        """
        return [
            DeclWithDependencies.from_dict(decl_data)
            for decl_data in analysis_dict.get('declarations', [])
        ]

    @staticmethod
    def load_from_file(file_path: str) -> 'DeclWithDependencies':
        raise NotImplementedError("load_from_file must be implemented by the child class")
    
    @staticmethod
    def load_from_string(json_text: str) -> 'DeclWithDependencies':
        raise NotImplementedError("load_from_string must be implemented by the child class")

class FileDependencyAnalysis(BaseModel):
    """
    File-level dependency analysis output from the Lean dependency parser.

    This model is used to parse the JSON output from the dependency-parser executable.
    For merging into larger collections, extract the declarations list.
    """
    file_path: str
    module_name: str
    imports: List[Dict]  # Raw import info
    declarations: List[DeclWithDependencies]

    def __repr__(self) -> str:
        return f"FileDependencyAnalysis({self.module_name}, {len(self.declarations)} decls)"

# Create an enum for parsing request type
class RequestType(Enum):
    PARSE_TACTICS = "parse_tactics"
    PARSE_THEOREM = "parse_theorem"
    CHKPT_TACTICS = "chkpt_tactics"
    BREAK_CHCKPNT = "break_chckpnt"
    PARSE_DEPENDS = "parse_depends"  # 13 chars - for dependency analysis

def get_path_to_tactic_parser_project() -> str:
    """Get the path to the tactic parser project directory."""
    tools_dir = os.path.dirname(__file__)
    tactic_parser_path = os.path.join(tools_dir, "tactic_parser")
    abs_path = os.path.abspath(tactic_parser_path)
    return abs_path

def get_path_to_tactic_parser_executable() -> str:
    """Get the path to the tactic parser executable."""
    abs_path = get_path_to_tactic_parser_project()
    tactic_parser_bin_path = os.path.join(abs_path, ".lake", "build", "bin", "tactic-parser")
    return tactic_parser_bin_path

def is_tactic_parser_built() -> bool:
    """Check if the tactic parser executable exists."""
    path_to_exec = get_path_to_tactic_parser_executable()
    if not os.path.isfile(path_to_exec):
        return False
    else:
        lean_version_needed = os.getenv("LEAN_VERSION", None)
        if lean_version_needed is None:
            return True
        tactic_parser_project = get_path_to_tactic_parser_project()
        # Check the version of the built parser
        toolchain_file = os.path.join(tactic_parser_project, "lean-toolchain")
        assert os.path.isfile(toolchain_file), f"lean-toolchain file not found at {toolchain_file}, something is wrong."
        with open(toolchain_file, 'r') as f:
            toolchain_content = f.read()
        toolchain_content = toolchain_content.strip()
        if toolchain_content.endswith(lean_version_needed):
            return True
        else:
            # Replace the version in the toolchain file
            # The version should be like 4.x.y
            pattern = r'^4\.\d+\.\d+$'
            if not re.match(pattern, lean_version_needed):
                raise RuntimeError(f"Tactic parser built with Lean version {toolchain_content}, but version {lean_version_needed} is required." +
                "Don't know how to build Lean which is not of the form 4.x.y. " +
                "Please rebuild the tactic parser.")
            toolchain_final = f"leanprover/lean4:v{lean_version_needed}"
            with open(toolchain_file, 'w') as f:
                f.write(toolchain_final)
            return False

def build_lean4_project(project_folder, logger: Optional[logging.Logger] = None, has_executable: bool = False):
    """Build the Lean4 project at the given folder."""

    logger = logger if logger else logging.getLogger(__name__)
    lake_folder = os.path.join(project_folder, ".lake")
    if os.path.exists(lake_folder):
        logger.info(f"Cleaning existing .lake folder at {lake_folder} before build.")
        shutil.rmtree(lake_folder)
    # Define the command
    if has_executable:
        command = f"cd {project_folder} && lake build"
    else:
        command = f"cd {project_folder} && lake exe cache get && lake build"
    
    logging.info(f"Building Lean4 project {project_folder}...")

    # Run the command
    # - shell=True is needed to process 'cd' and '&&'
    # - capture_output=True captures stdout and stderr
    # - text=True decodes stdout/stderr as text (using default encoding)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the build logs from stdout
    logging.info('-'*15 + f'Build Logs from {project_folder}' + '-'*15)
    logging.info(result.stdout)
    
    # Optionally print error logs if any exist
    if result.stderr:
        logging.error('-'*15 + f'Error Logs from {project_folder}' + '-'*15)
        logging.error(result.stderr)

    logging.info('-'*15 + f'End Build Logs from {project_folder}' + '-'*15)

    # --- Here is how you check the exit code ---
    exit_code = result.returncode
    logging.info(f"Process finished with exit code: {exit_code}")

    # You can now act on the exit code
    if exit_code == 0:
        logging.info("Build successful!")
    else:
        logging.error("Build FAILED!")
        raise Exception(f"Build failed with code {exit_code}")

def build_tactic_parser_if_needed(logger: Optional[logging.Logger] = None):
    """Build the tactic parser if not already built."""
    if not is_tactic_parser_built():
        build_lean4_project(get_path_to_tactic_parser_project(), logger, has_executable=True)

def get_path_to_dependency_parser_executable() -> str:
    """Get the path to the dependency parser executable."""
    abs_path = get_path_to_tactic_parser_project()
    dependency_parser_bin_path = os.path.join(abs_path, ".lake", "build", "bin", "dependency-parser")
    return dependency_parser_bin_path

def analyze_lean_file_dependencies(
    full_lean_file_path: str,
    json_output_path: str,
    working_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> tuple[List[FileDependencyAnalysis], List[ErrorInfo]]:
    """
    Analyze dependencies in a Lean file and export to JSON.

    Args:
        full_lean_file_path: Path to the Lean file to analyze (relative to working_dir)
        json_output_path: Path where JSON output will be written (relative to working_dir)
        working_dir: Working directory (Lean project root). If None, uses current directory.
        logger: Optional logger for debugging

    Returns:
        tuple: (FileDependencyAnalysis, List[ErrorInfo]) - analysis results and any errors

    Raises:
        FileNotFoundError: If the executable or input file doesn't exist
        subprocess.CalledProcessError: If the analysis fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if working_dir is None:
        working_dir = os.getcwd()

    # Ensure the dependency parser is built
    build_tactic_parser_if_needed(logger)

    # Get the executable path
    exec_path = get_path_to_dependency_parser_executable()
    if not os.path.isfile(exec_path):
        raise FileNotFoundError(
            f"Dependency parser executable not found at {exec_path}. "
            "Please build it first with 'lake build'"
        )

    # Verify the input file exists (relative to working_dir)
    # full_lean_path = Path(working_dir) / lean_file_path
    if not os.path.exists(full_lean_file_path):
        raise FileNotFoundError(f"Lean file not found: {full_lean_file_path}")

    # Build the command
    cmds = ["lake", "env", str(exec_path), full_lean_file_path, json_output_path]

    logger.debug(f"Running dependency analysis: {' '.join(cmds)}")
    logger.debug(f"Working directory: {working_dir}")

    # Execute the command
    result = subprocess.run(
        cmds,
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False  # Don't raise on error, handle it ourselves
    )

    logger.debug(f"Dependency analysis stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"Dependency analysis stderr: {result.stderr}")

    # Check for errors
    errors: List[ErrorInfo] = []
    if result.returncode != 0 or result.stderr:
        error_msg = f"Dependency parser failed with code {result.returncode}"
        if result.stderr:
            error_msg += f": {result.stderr}"
        errors.append(ErrorInfo(message=error_msg, position=Position(line=0, column=0)))

    # Read and parse the JSON output
    full_json_path = Path(working_dir) / json_output_path
    if full_json_path.exists():
        with open(full_json_path, 'r') as f:
            data = json.load(f)
        analysis = FileDependencyAnalysis.model_validate(data)
    else:
        # Create empty analysis if file doesn't exist
        analysis = FileDependencyAnalysis(
            file_path=full_lean_file_path,
            module_name="",
            imports=[],
            declarations=[]
        )
        if not errors:
            errors.append(ErrorInfo(
                message="Output file was not created",
                position=Position(line=0, column=0)
            ))

    return [analysis], errors

def get_from_original_text(code: str, lean_info: LeanLineInfo, relative_line_num : int = 1) -> str:
    """Extract the text corresponding to a LeanLineInfo from the code."""
    lines = code.splitlines()
    start_line_idx = lean_info.line - relative_line_num
    end_line_idx = lean_info.end_line - relative_line_num

    if start_line_idx < 0 or end_line_idx >= len(lines):
        raise ValueError("LeanLineInfo line numbers are out of bounds")

    if start_line_idx == end_line_idx:
        # Single line case
        return lines[start_line_idx][lean_info.column:lean_info.end_column]
    else:
        # Multi-line case
        extracted_lines = []
        # First line
        extracted_lines.append(lines[start_line_idx][lean_info.column:])
        # Middle lines
        for i in range(start_line_idx + 1, end_line_idx):
            extracted_lines.append(lines[i])
        # Last line
        extracted_lines.append(lines[end_line_idx][:lean_info.end_column])
        return '\n'.join(extracted_lines)

theorem_name_regex = r"(((theorem|lemma)[\s]+([^\s:]*))|example)"
theorem_name_match = re.compile(theorem_name_regex, re.MULTILINE)

def parse_theorem_name(thm_stmt: str) -> Optional[str]:
    match = theorem_name_match.search(thm_stmt)
    if match:
        theorem_name = match.group(4)
        return theorem_name
    return None

class TacticParser:
    """Parse tactics from Lean 4 code without compilation.

    The parser process runs in the background and is reused across multiple requests.

    If you want to parse tactics that use mathlib or other dependencies, provide a
    project_path when initializing the parser. The process will run from that directory
    and automatically find the project's .lake/build with all dependencies.
    """

    def __init__(self, parser_path: Optional[str] = None, project_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
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
            # Ensure the parser is built
            build_tactic_parser_if_needed(self.logger)
            path_to_tactic_parser_exec = get_path_to_tactic_parser_executable()
            assert os.path.isfile(path_to_tactic_parser_exec), f"Tactic parser executable not found at {path_to_tactic_parser_exec}, please build it first."
            cmds = ["lake", "env", path_to_tactic_parser_exec]
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

    def _is_tactic_request(self, parse_type: RequestType) -> bool:
        return parse_type == RequestType.PARSE_TACTICS or parse_type == RequestType.CHKPT_TACTICS or parse_type == RequestType.BREAK_CHCKPNT

    def parse(self, lean_code: str, fail_on_error: bool = True, parse_type: RequestType = RequestType.PARSE_TACTICS) -> tuple[List[LeanLineInfo], List[ErrorInfo]]:
        """
        Parse tactics from Lean 4 code.

        Args:
            lean_code: Lean 4 source code as a string

        Returns:
            List of leanInfo objects

        Raises:
            RuntimeError: If parsing fails
        """
        self._ensure_running()

        retry_cnt = 5
        succeeded = False

        while retry_cnt > 0 and not succeeded:
            # Encode Lean code as base64
            final_code = parse_type.value + lean_code
            b64_input = base64.b64encode(final_code.encode('utf-8')).decode('ascii')
            self.logger.debug(f"Sending {len(final_code)} bytes of Lean code")
            self.logger.debug(f"Base64 encoded input length: {len(b64_input)}")
            self.logger.debug(f"Input (base64): {b64_input}")

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
        errors : List[ErrorInfo] = []
        if response.get("errors"):
            if fail_on_error:
                raise RuntimeError(f"Parse error: {response['errors']}")
            else:
                for err in response["errors"]:
                    error_info = ErrorInfo.model_validate(err)
                    errors.append(error_info)
                    self.logger.debug(f"Parse error: {error_info}")

        # Convert tree to leanInfo objects
        trees : list[TreeNode] = []
        tactics = []
        for t in response.get("trees", []):
            if t is not None:
                tree = TreeNode.model_validate(t)
                trees.append(tree)
        for t in trees:
            assert t.start_pos is not None
            assert t.end_pos is not None
            if t.decl_type is not None and (t.decl_type == "theorem" or t.decl_type == "lemma"):
                # TODO: Fix the incorrect theorem/lemma name parsing from the underlying lean tool
                actual_name = parse_theorem_name(t.text if t.text else "")
                assert actual_name is not None, "Theorem/lemma name should not be None"
                if t.name != actual_name:
                    t.name = actual_name
            tactics.append(
                LeanLineInfo(
                    text=t.text if t.text else "",
                    line=t.start_pos.line,
                    column=t.start_pos.column,
                    end_line=t.end_pos.line,
                    end_column=t.end_pos.column,
                    decl_type=t.decl_type,
                    name=t.name,
                    doc_string=t.doc_string,
                    namespace=t.namespace
                )
            )
        self.logger.debug(f"Parsed {len(tactics)} tactics")

        return tactics, errors

    def parse_file(self, file_path: str, parse_type: RequestType = RequestType.PARSE_THEOREM, json_output_path: Optional[str] = None) -> tuple[Union[List[LeanLineInfo], List[FileDependencyAnalysis]], List[ErrorInfo]]:
        """
        Parse tactics from a Lean 4 file or analyze its dependencies.

        Args:
            file_path: Path to the Lean 4 file
            parse_type: Type of parsing to perform
            json_output_path: For PARSE_DEPENDS only - path where JSON output will be written.
                            If None, generates a temporary path.

        Returns:
            - For PARSE_TACTICS/PARSE_THEOREM/CHKPT_TACTICS/BREAK_CHCKPNT: tuple of (List[LeanLineInfo], List[ErrorInfo])
            - For PARSE_DEPENDS: tuple of (FileDependencyAnalysis, List[ErrorInfo])
        """
        if parse_type == RequestType.PARSE_DEPENDS:
            # Use dependency parser executable
            if json_output_path is None:
                # Generate a temporary output path
                json_output_file_path = Path(file_path).with_suffix('.deps.json')
                json_output_path = str(json_output_file_path)

            # Determine working directory
            if self.project_path:
                working_dir = self.project_path
                # Make file_path relative to working_dir if it's absolute
                file_path_obj = Path(file_path)
                # Make sure that path is absolute
                if not file_path_obj.is_absolute():
                    file_path = str(file_path_obj.resolve())
                working_dir = str(Path(working_dir).resolve())
                json_output_path = str(Path(json_output_path).resolve())
            else:
                working_dir = str(Path(file_path).parent.resolve())
                file_path = str(Path(file_path).resolve())
                json_output_path = str(Path(json_output_path).resolve())

            return analyze_lean_file_dependencies(
                full_lean_file_path=file_path,
                json_output_path=json_output_path,
                working_dir=working_dir,
                logger=self.logger
            )
        else:
            # Use normal tactic parser
            with open(file_path, 'r', encoding='utf-8') as f:
                lean_code = f.read()
            return self.parse(lean_code, parse_type=parse_type)

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
def print_tactics(tactics: List[LeanLineInfo], logger: Optional[logging.Logger] = None):
    for tactic in tactics:
        msg = f"Line {tactic.line}, Col {tactic.column} to Line {tactic.end_line}, Col {tactic.end_column}: {tactic.text}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    project_path = str(Path(__file__).parent.parent.parent / "data" / "test" / "lean4_proj")

    with TacticParser() as parser:
        # Example 1: Simple proof
        lean_code = "example : True := by trivial"

        print("Parsing example 1...")
        tactics, errors = parser.parse(lean_code)
        print_tactics(tactics)
        if errors:
            print(f"Error: {errors}")
    p_path = "/home/amthakur/Projects/copra/data/test/miniF2F-lean4"
    with TacticParser(project_path=p_path) as parser:
        # Example 1a: Simple proof with multiple tactics
        lean_code = """
import MiniF2F.Minif2fImport
open BigOperators Real Nat Topology

theorem mathd_algebra_33
  (x y z : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 2 * x = 5 * y)
  (h₂ : 7 * y = 10 * z) :
  z / x = 7 / 25 :=
by
have h1': x = 5 * y / 2 := by ring
"""
        print("Parsing example 1a...")
        tactics, errors = parser.parse(lean_code, fail_on_error=False)
        print_tactics(tactics)
        if errors:
            print(f"Error: {errors}")

    with TacticParser() as parser:
        # Example 1b: Simple have proofs
        # p \implies q and q \implies r then have p \implies r
        lean_code = """
example (p q r: Prop) (h1: p → q) (h2: q → r) : p → r := by
    have h3: p → r := 
    by
        try simp
        wrong_tactic
"""

        print("Parsing example 1b...")
        tactics, errors = parser.parse(lean_code, fail_on_error=False)
        print_tactics(tactics)
        if errors:
            print(f"Error: {errors}")



    with TacticParser(project_path=project_path) as parser:
        # Example 2: Multiline with params
        lean_code2 = "example (r: Nat) (p q : Prop) (hp : p) (hq : q) : p ∧ q := by\n  apply And.intro\n  exact hp\n  exact hq"

        print("\nParsing example 2...")
        tactics2, errors = parser.parse(lean_code2)
        print_tactics(tactics2)
        if errors:
            print(f"Error: {errors}")
        
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
        tactics3, errors = parser.parse(lean_code3)
        print_tactics(tactics3)
        if errors:
            print(f"Error: {errors}")
    
    file_path = str(Path(__file__).parent.parent.parent / "data" / "test" / "lean4_proj" / "Lean4Proj" / "Basic.lean")

    with TacticParser(project_path=project_path) as parser:
        # Example 4: Parse from file
        print("\nParsing example 4 (from file)...")
        tactics4, errors = parser.parse_file(file_path)
        print_tactics(tactics4)
        if errors:
            print(f"Error: {errors}")

    with TacticParser(project_path=project_path) as parser:
        # Example 2: Multiline with params
        lean_code4 = "example (r: ℕ) (p q : Prop) (hp : p) (hq : q) : p ∧ q := by grind"

        print("\nParsing example 5...")
        tactics5, errors = parser.parse(lean_code4)
        print_tactics(tactics5)
        if errors:
            print(f"Error: {errors}")
    
    with TacticParser(project_path=project_path) as parser:
        # Example 6: Parse tactics from file with multiple theorems
        print("\nParsing example 6 (theorem parsing from file)...")
        tactics6, errors = parser.parse(lean_code3 + "\n" + lean_code4, parse_type=RequestType.PARSE_TACTICS)
        print_tactics(tactics6)
        if errors:
            print(f"Error: {errors}")
        
    with TacticParser(project_path=project_path) as parser:
        # Example 7: Parse tactics which are wrong
        print("\nParsing example 7 (theorem declaration parsing from file)...")
        lean_code5 = "theorem wrong_decl : Nat := by assdfadfs"
        tactics7, errors = parser.parse(lean_code5, fail_on_error=False)
        print_tactics(tactics7)
        if errors:
            print(f"Error: {errors}")
    
    with TacticParser(project_path=project_path) as parser:
        # Example 8: Parse tactics just before `by`
        print("\nParsing example 8 (theorem with just before `by`...)")
        lean_code8 = "theorem temp: 1 + 2 = 3 :=\nby"
        tactics8, errors = parser.parse(lean_code8, fail_on_error=False)
        print_tactics(tactics8)
        if errors:
            print(f"Error: {errors}")
    
    with TacticParser(project_path=project_path) as parser:
        # Example 9: Parse tactics just before `by`
        print("\nParsing example 9 (theorem with just before `by`...)")
        lean_code9 = "import Mathlib\ntheorem temp: 1 + 2 = 3 :=\nby\n    have h1: 1 + 1 = 2 := by\n        linarith\n        done"
        tactics9, errors = parser.parse(lean_code9, fail_on_error=False)
        print_tactics(tactics9)
        if errors:
            print(f"Error: {errors}")
    
    with TacticParser(project_path=project_path) as parser:
        # Example 10: Test checkpointing
        print("\nParsing example 10 (checkpointing...)")
        lean_code10 = """import Mathlib

theorem temp: 1 + 2 = 3 :=
by
linarith

theorem temp1: 3 = 1 + 1 + 1 :=
by
linarith
"""
        tactics10, errors = parser.parse(lean_code10, fail_on_error=True, parse_type=RequestType.CHKPT_TACTICS)
        print_tactics(tactics10)
        if errors:
            print(f"Error: {errors}")
        # Now just execute from the checkpoint
        lean_code10b = """
theorem temp2: 1 + 2 = 3 :=
by
have h_temp := temp1
"""
        print("\nContinuing from checkpoint...")
        tactics10b, errors = parser.parse(lean_code10b, fail_on_error=False, parse_type=RequestType.PARSE_TACTICS)
        print_tactics(tactics10b)
        if errors:
            # The error should contain h_temp
            print(f"Error: {errors}")
        
        print("\nBreaking checkpoint...")
        new_lean_code10c = lean_code10 + lean_code10b
        tactics10c, errors = parser.parse(new_lean_code10c, fail_on_error=False, parse_type=RequestType.BREAK_CHCKPNT)
        # ^This will reimport everything all run all theorems from scratch
        print_tactics(tactics10c)
        if errors:
            print(f"Error: {errors}")