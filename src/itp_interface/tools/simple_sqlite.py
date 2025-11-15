"""
Simple SQLite database for storing and querying Lean declaration data.

This module provides a thread-safe SQLite database for storing Lean declarations,
their dependencies, and file metadata. Designed to work with Ray actors processing
files in parallel.
"""

import sqlite3
import uuid
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class LeanDeclarationDB:
    """
    Thread-safe SQLite database for storing Lean file and declaration information.

    Key features:
    - Automatic ID assignment on first discovery (declaration or dependency)
    - Simplified dependency storage (edges only: A depends on B)
    - Thread-safe operations with WAL mode for concurrent Ray actors
    - Idempotent operations for safe parallel execution
    """

    def __init__(self, db_path: str, timeout: float = 30.0):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
            timeout: Timeout in seconds for database locks (default 30s)
        """
        self.db_path = db_path
        self.timeout = timeout
        self.conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        # Explicitly set UTF-8 encoding
        self.conn.execute("PRAGMA encoding = 'UTF-8'")
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode BEFORE creating tables for better concurrency
        self.enable_wal_mode()
        # Create tables (safe with IF NOT EXISTS even from multiple actors)
        self._create_tables()

    def _generate_unique_id(self) -> str:
        """
        Generate a unique ID for a declaration.

        Format: {timestamp}_{uuid4}
        Same format as used in lean4_local_data_extraction_transform.py

        Returns:
            Unique identifier string
        """
        timestamp = str(int(uuid.uuid1().time_low))
        random_id = str(uuid.uuid4())
        return f"{timestamp}_{random_id}"

    def enable_wal_mode(self):
        """
        Enable Write-Ahead Logging mode for better concurrent write performance.

        WAL mode allows multiple readers and one writer to access the database
        simultaneously, which is essential for Ray actors processing files in parallel.
        """
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.commit()

    def _create_tables(self):
        """
        Create the database schema with proper indexes and constraints.

        Safe for concurrent execution from multiple Ray actors:
        - Uses CREATE TABLE IF NOT EXISTS (idempotent)
        - Uses CREATE INDEX IF NOT EXISTS (idempotent)
        - WAL mode is enabled before this is called
        - 30s timeout handles lock contention
        """
        cursor = self.conn.cursor()

        # Files table - stores file metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                module_name TEXT NOT NULL
            )
        """)

        # Imports table - stores file import relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                end_pos INTEGER,
                module_name TEXT,
                start_pos INTEGER,
                text TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # Declarations table - stores all declarations (complete or partial)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS declarations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decl_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                namespace TEXT,
                file_path TEXT,
                module_name TEXT,
                decl_type TEXT,
                text TEXT,
                line INTEGER,
                column INTEGER,
                end_line INTEGER,
                end_column INTEGER,
                doc_string TEXT,
                proof TEXT,
                -- A declaration is uniquely identified by its name and location
                UNIQUE(name, namespace, file_path, module_name)
            )
        """)

        # Simplified dependencies table - stores only edges (A depends on B)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS declaration_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_decl_id TEXT NOT NULL,
                to_decl_id TEXT NOT NULL,
                UNIQUE(from_decl_id, to_decl_id),
                FOREIGN KEY (from_decl_id) REFERENCES declarations(decl_id) ON DELETE CASCADE,
                FOREIGN KEY (to_decl_id) REFERENCES declarations(decl_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_files_path
            ON files(file_path)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_files_module
            ON files(module_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_declarations_name
            ON declarations(name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_declarations_namespace
            ON declarations(namespace)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_declarations_decl_id
            ON declarations(decl_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_declarations_lookup
            ON declarations(name, namespace, file_path, module_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dependencies_from
            ON declaration_dependencies(from_decl_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dependencies_to
            ON declaration_dependencies(to_decl_id)
        """)

        self.conn.commit()

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            with db.transaction():
                db.insert_something(...)
                db.insert_something_else(...)
        """
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def get_or_create_decl_id(
        self,
        name: str,
        namespace: Optional[str] = None,
        file_path: Optional[str] = None,
        module_name: Optional[str] = None
    ) -> str:
        """
        Get existing decl_id or create a new one for a declaration.

        This is the core method for ID assignment. IDs are assigned as soon as
        a declaration is discovered (either as a dependency or as a declaration itself).

        Args:
            name: Declaration name (required)
            namespace: Namespace (can be None)
            file_path: File path (can be None for unresolved dependencies)
            module_name: Module name (can be None for unresolved dependencies)

        Returns:
            The decl_id (existing or newly created)
        """
        cursor = self.conn.cursor()

        # Try to find existing declaration
        # Handle NULL values properly in SQL
        cursor.execute("""
            SELECT decl_id FROM declarations
            WHERE name = ?
              AND (namespace IS ? OR (namespace IS NULL AND ? IS NULL))
              AND (file_path IS ? OR (file_path IS NULL AND ? IS NULL))
              AND (module_name IS ? OR (module_name IS NULL AND ? IS NULL))
        """, (name, namespace, namespace, file_path, file_path, module_name, module_name))

        row = cursor.fetchone()
        if row:
            return row[0]  # Return existing ID

        # Generate new ID and insert minimal record
        new_decl_id = self._generate_unique_id()

        try:
            cursor.execute("""
                INSERT INTO declarations (decl_id, name, namespace, file_path, module_name)
                VALUES (?, ?, ?, ?, ?)
            """, (new_decl_id, name, namespace, file_path, module_name))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Race condition: another process inserted it between our SELECT and INSERT
            # Query again to get the existing ID
            cursor.execute("""
                SELECT decl_id FROM declarations
                WHERE name = ?
                  AND (namespace IS ? OR (namespace IS NULL AND ? IS NULL))
                  AND (file_path IS ? OR (file_path IS NULL AND ? IS NULL))
                  AND (module_name IS ? OR (module_name IS NULL AND ? IS NULL))
            """, (name, namespace, namespace, file_path, file_path, module_name, module_name))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                # This shouldn't happen, but raise if it does
                raise

        return new_decl_id

    def upsert_declaration_full_info(
        self,
        decl_id: str,
        name: str,
        namespace: Optional[str],
        file_path: str,
        module_name: str,
        decl_type: Optional[str] = None,
        text: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        doc_string: Optional[str] = None,
        proof: Optional[str] = None
    ):
        """
        Update a declaration with complete information.

        This is called when we process the actual declaration (not just a reference).
        Updates the record with full metadata.

        Args:
            decl_id: The declaration ID
            name: Declaration name
            namespace: Namespace
            file_path: File path
            module_name: Module name
            decl_type: Declaration type (theorem, def, etc.)
            text: Full declaration text
            line: Starting line number
            column: Starting column number
            end_line: Ending line number
            end_column: Ending column number
            doc_string: Documentation string
            proof: Proof text
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE declarations
            SET decl_type = ?,
                text = ?,
                line = ?,
                column = ?,
                end_line = ?,
                end_column = ?,
                doc_string = ?,
                proof = ?
            WHERE decl_id = ?
        """, (decl_type, text, line, column, end_line, end_column, doc_string, proof, decl_id))

        self.conn.commit()

    def insert_dependency_edge(self, from_decl_id: str, to_decl_id: str):
        """
        Insert a dependency edge: from_decl_id depends on to_decl_id.

        Uses INSERT OR IGNORE for idempotency (safe to call multiple times).

        Args:
            from_decl_id: The declaration that has the dependency
            to_decl_id: The declaration being depended on
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO declaration_dependencies (from_decl_id, to_decl_id)
            VALUES (?, ?)
        """, (from_decl_id, to_decl_id))

        self.conn.commit()

    def process_fda_list(self, fda_list: List) -> List[str]:
        """
        Process a list of FileDependencyAnalysis objects.

        Args:
            fda_list: List of FileDependencyAnalysis objects

        Returns:
            List of all decl_ids that were processed
        """
        all_decl_ids = []

        for fda in fda_list:
            # Insert file and imports first
            if fda.imports:
                self.insert_file_imports(fda.file_path, fda.module_name, fda.imports)

            # Process each declaration in this file
            for decl in fda.declarations:
                decl_id = self.process_declaration(fda.file_path, fda.module_name, decl)
                all_decl_ids.append(decl_id)

        return all_decl_ids

    def process_declaration(
        self,
        fda_file_path: str,
        fda_module_name: str,
        decl
    ) -> str:
        """
        Process a declaration from a FileDependencyAnalysis object.

        This is the main high-level method that:
        1. Gets or creates decl_id for this declaration
        2. Updates full declaration info
        3. Processes all dependencies and creates edges

        Args:
            fda_file_path: File path from FileDependencyAnalysis
            fda_module_name: Module name from FileDependencyAnalysis
            decl: DeclWithDependencies object

        Returns:
            The decl_id for this declaration
        """
        # Get or create ID for this declaration
        decl_id = self.get_or_create_decl_id(
            name=decl.decl_info.name,
            namespace=decl.decl_info.namespace,
            file_path=fda_file_path,
            module_name=fda_module_name
        )

        # Update with full declaration info
        self.upsert_declaration_full_info(
            decl_id=decl_id,
            name=decl.decl_info.name,
            namespace=decl.decl_info.namespace,
            file_path=fda_file_path,
            module_name=fda_module_name,
            decl_type=decl.decl_info.decl_type,
            text=decl.decl_info.text,
            line=decl.decl_info.line,
            column=decl.decl_info.column,
            end_line=decl.decl_info.end_line,
            end_column=decl.decl_info.end_column,
            doc_string=decl.decl_info.doc_string,
            proof=decl.decl_info.proof
        )

        # Process dependencies
        for dep in decl.dependencies:
            # Get or create ID for the dependency
            dep_decl_id = self.get_or_create_decl_id(
                name=dep.name,
                namespace=dep.namespace,
                file_path=dep.file_path,
                module_name=dep.module_name
            )

            # Propagate the decl_id back to the dependency object
            dep.decl_id = dep_decl_id

            # Insert dependency edge
            self.insert_dependency_edge(decl_id, dep_decl_id)

        return decl_id

    def get_or_create_file(self, file_path: str, module_name: str) -> int:
        """
        Get or create a file record.

        Args:
            file_path: The file path
            module_name: The module name

        Returns:
            The file_id (existing or newly created)
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute("""
            INSERT INTO files (file_path, module_name)
            VALUES (?, ?)
        """, (file_path, module_name))
        self.conn.commit()

        file_id = cursor.lastrowid
        if file_id is None:
            raise RuntimeError("Failed to get file_id after insert")
        return file_id

    def insert_file_imports(self, file_path: str, module_name: str, imports: List[Dict]):
        """
        Insert imports for a file.

        Args:
            file_path: The file path
            module_name: The module name
            imports: List of import dictionaries
        """
        file_id = self.get_or_create_file(file_path, module_name)
        cursor = self.conn.cursor()

        for import_data in imports:
            cursor.execute("""
                INSERT INTO imports (file_id, end_pos, module_name, start_pos, text)
                VALUES (?, ?, ?, ?, ?)
            """, (file_id, import_data.get('end_pos'), import_data.get('module_name'),
                  import_data.get('start_pos'), import_data.get('text')))

        self.conn.commit()

    # Query methods

    def get_declaration_by_decl_id(self, decl_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a declaration by its decl_id.

        Args:
            decl_id: The unique declaration ID

        Returns:
            Dictionary containing declaration information or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM declarations WHERE decl_id = ?", (decl_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_declarations_by_name(
        self,
        name: str,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get declarations by name and optionally namespace.

        Args:
            name: The declaration name
            namespace: Optional namespace to filter by

        Returns:
            List of dictionaries containing declaration information
        """
        cursor = self.conn.cursor()

        if namespace is not None:
            cursor.execute("""
                SELECT * FROM declarations
                WHERE name = ? AND namespace = ?
            """, (name, namespace))
        else:
            cursor.execute("""
                SELECT * FROM declarations
                WHERE name = ?
            """, (name,))

        return [dict(row) for row in cursor.fetchall()]

    def get_declarations_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get all declarations in a specific file.

        Args:
            file_path: The file path

        Returns:
            List of dictionaries containing declaration information
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM declarations
            WHERE file_path = ?
            ORDER BY line
        """, (file_path,))

        return [dict(row) for row in cursor.fetchall()]

    def get_declarations_by_module(self, module_name: str) -> List[Dict[str, Any]]:
        """
        Get all declarations in a specific module.

        Args:
            module_name: The module name

        Returns:
            List of dictionaries containing declaration information
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM declarations
            WHERE module_name = ?
        """, (module_name,))

        return [dict(row) for row in cursor.fetchall()]

    def get_dependencies(self, decl_id: str) -> List[Dict[str, Any]]:
        """
        Get all declarations that this declaration depends on.

        Args:
            decl_id: The declaration ID

        Returns:
            List of declarations that this declaration depends on
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.* FROM declarations d
            JOIN declaration_dependencies dd ON dd.to_decl_id = d.decl_id
            WHERE dd.from_decl_id = ?
        """, (decl_id,))

        return [dict(row) for row in cursor.fetchall()]

    def get_dependents(self, decl_id: str) -> List[Dict[str, Any]]:
        """
        Get all declarations that depend on this declaration.

        Args:
            decl_id: The declaration ID

        Returns:
            List of declarations that depend on this one
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.* FROM declarations d
            JOIN declaration_dependencies dd ON dd.from_decl_id = d.decl_id
            WHERE dd.to_decl_id = ?
        """, (decl_id,))

        return [dict(row) for row in cursor.fetchall()]

    def get_dependency_graph(
        self,
        decl_id: str,
        max_depth: int = 10,
        direction: str = 'dependencies'
    ) -> Dict[str, Any]:
        """
        Get the dependency graph for a declaration (recursive).

        Args:
            decl_id: The declaration ID to start from
            max_depth: Maximum depth to traverse (default 10)
            direction: 'dependencies' (what this depends on) or 'dependents' (what depends on this)

        Returns:
            Dictionary containing the dependency graph
        """
        visited = set()

        def _get_graph_recursive(current_decl_id: str, depth: int) -> Optional[Dict[str, Any]]:
            if depth > max_depth or current_decl_id in visited:
                return None

            visited.add(current_decl_id)

            decl = self.get_declaration_by_decl_id(current_decl_id)
            if not decl:
                return None

            if direction == 'dependencies':
                related = self.get_dependencies(current_decl_id)
            else:
                related = self.get_dependents(current_decl_id)

            result = {
                'declaration': decl,
                direction: []
            }

            for rel_decl in related:
                sub_graph = _get_graph_recursive(rel_decl['decl_id'], depth + 1)
                if sub_graph:
                    result[direction].append(sub_graph)

            return result

        graph = _get_graph_recursive(decl_id, 0)
        return graph if graph else {}

    def search_declarations(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        file_path: Optional[str] = None,
        module_name: Optional[str] = None,
        decl_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search declarations with multiple optional filters.

        Args:
            name: Declaration name (optional)
            namespace: Namespace (optional)
            file_path: File path (optional)
            module_name: Module name (optional)
            decl_type: Declaration type (optional)

        Returns:
            List of matching declarations
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM declarations WHERE 1=1"
        params = []

        if name is not None:
            query += " AND name = ?"
            params.append(name)

        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)

        if file_path is not None:
            query += " AND file_path = ?"
            params.append(file_path)

        if module_name is not None:
            query += " AND module_name = ?"
            params.append(module_name)

        if decl_type is not None:
            query += " AND decl_type = ?"
            params.append(decl_type)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.

        Returns:
            Dictionary with counts of files, declarations, and dependencies
        """
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM files")
        stats['total_files'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM declarations")
        stats['total_declarations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM declaration_dependencies")
        stats['total_dependencies'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM declarations WHERE file_path IS NULL")
        stats['unresolved_declarations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM imports")
        stats['total_imports'] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


if __name__ == "__main__":
    # Example usage
    print("Creating test database...")

    with LeanDeclarationDB("test_lean_declarations.db") as db:
        # Test with file and imports
        print("\nTesting file and imports insertion:")
        test_imports = [
            {
                "end_pos": 292,
                "module_name": "Mathlib.Algebra.Group.Basic",
                "start_pos": 258,
                "text": "import Mathlib.Algebra.Group.Basic"
            },
            {
                "end_pos": 321,
                "module_name": "Mathlib.Tactic.Common",
                "start_pos": 293,
                "text": "import Mathlib.Tactic.Common"
            }
        ]

        db.insert_file_imports(
            file_path="Mathlib/Algebra/Divisibility/Basic.lean",
            module_name="Mathlib.Algebra.Divisibility.Basic",
            imports=test_imports
        )
        print("Inserted file and imports")

        # Test ID generation
        print("\nTesting ID generation:")
        decl_id1 = db.get_or_create_decl_id(
            name="dvd_trans",
            namespace=None,
            file_path="Mathlib/Algebra/Divisibility/Basic.lean",
            module_name="Mathlib.Algebra.Divisibility.Basic"
        )
        print(f"Generated ID for dvd_trans: {decl_id1}")

        # Get same ID again (should return existing)
        decl_id2 = db.get_or_create_decl_id(
            name="dvd_trans",
            namespace=None,
            file_path="Mathlib/Algebra/Divisibility/Basic.lean",
            module_name="Mathlib.Algebra.Divisibility.Basic"
        )
        print(f"Retrieved ID for dvd_trans: {decl_id2}")
        print(f"IDs match: {decl_id1 == decl_id2}")

        # Update with full info
        db.upsert_declaration_full_info(
            decl_id=decl_id1,
            name="dvd_trans",
            namespace=None,
            file_path="Mathlib/Algebra/Divisibility/Basic.lean",
            module_name="Mathlib.Algebra.Divisibility.Basic",
            decl_type="theorem",
            text="@[trans]\ntheorem dvd_trans : a # b � b # c � a # c",
            line=63,
            column=0,
            end_line=68,
            end_column=0,
            doc_string=None,
            proof="| �d, h��, �e, h�� => �d * e, h� � h�.trans <| mul_assoc a d e�"
        )
        print("Updated declaration with full info")

        # Create a dependency
        dep_id = db.get_or_create_decl_id(
            name="mul_assoc",
            namespace=None,
            file_path=None,
            module_name=None
        )
        print(f"\nGenerated ID for dependency mul_assoc: {dep_id}")

        # Insert dependency edge
        db.insert_dependency_edge(decl_id1, dep_id)
        print("Inserted dependency edge")

        # Query
        print("\nQuerying declaration:")
        decl = db.get_declaration_by_decl_id(decl_id1)
        if decl:
            print(f"Declaration: {decl['name']} ({decl['decl_type']})")
        else:
            print("Declaration not found!")

        print("\nQuerying dependencies:")
        deps = db.get_dependencies(decl_id1)
        print(f"Dependencies: {[d['name'] for d in deps]}")

        # Statistics
        print("\nDatabase statistics:")
        stats = db.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\nTest complete!")
