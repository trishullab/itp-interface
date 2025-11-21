"""
Database utility functions for the Lean Declaration DB Explorer.
"""

import sys
from pathlib import Path

# Add parent directories to path to import from itp_interface
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from typing import Dict, List, Any, Optional
from itp_interface.tools.simple_sqlite import LeanDeclarationDB


def execute_query(db: LeanDeclarationDB, query: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Args:
        db: LeanDeclarationDB instance
        query: SQL query string

    Returns:
        DataFrame with query results
    """
    cursor = db.conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()

    if rows:
        # Get column names
        columns = [description[0] for description in cursor.description]
        # Convert to DataFrame
        return pd.DataFrame([dict(row) for row in rows], columns=columns)
    else:
        return pd.DataFrame()


def get_common_queries() -> Dict[str, str]:
    """
    Return a dictionary of common pre-built queries.

    Returns:
        Dict mapping query name to SQL query string
    """
    return {
        "Show all files": """
            SELECT file_path, module_name
            FROM files
            ORDER BY file_path
        """,
        "Show all declarations": """
            SELECT name, namespace, decl_type, file_path, module_name
            FROM declarations
            WHERE file_path IS NOT NULL
            ORDER BY file_path, line
            LIMIT 100
        """,
        "Count declarations by type": """
            SELECT decl_type, COUNT(*) as count
            FROM declarations
            WHERE decl_type IS NOT NULL
            GROUP BY decl_type
            ORDER BY count DESC
        """,
        "Count declarations by file": """
            SELECT file_path, COUNT(*) as count
            FROM declarations
            WHERE file_path IS NOT NULL
            GROUP BY file_path
            ORDER BY count DESC
            LIMIT 20
        """,
        "Show unresolved declarations": """
            SELECT name, namespace
            FROM declarations
            WHERE file_path IS NULL
            LIMIT 100
        """,
        "Show imports for all files": """
            SELECT f.file_path, i.module_name as import_module, i.text
            FROM files f
            JOIN imports i ON i.file_id = f.id
            ORDER BY f.file_path
        """,
        "Show most depended-on declarations": """
            SELECT d.name, d.namespace, d.file_path, COUNT(*) as dependents_count
            FROM declarations d
            JOIN declaration_dependencies dd ON dd.to_decl_id = d.decl_id
            GROUP BY d.decl_id
            ORDER BY dependents_count DESC
            LIMIT 20
        """,
        "Show declarations with most dependencies": """
            SELECT d.name, d.namespace, d.file_path, COUNT(*) as dependencies_count
            FROM declarations d
            JOIN declaration_dependencies dd ON dd.from_decl_id = d.decl_id
            GROUP BY d.decl_id
            ORDER BY dependencies_count DESC
            LIMIT 20
        """
    }


def search_declarations(
    db: LeanDeclarationDB,
    name_pattern: Optional[str] = None,
    namespace: Optional[str] = None,
    file_path: Optional[str] = None,
    decl_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Search declarations with filters.

    Args:
        db: LeanDeclarationDB instance
        name_pattern: Name pattern (SQL LIKE syntax)
        namespace: Namespace filter
        file_path: File path filter
        decl_type: Declaration type filter

    Returns:
        DataFrame with matching declarations
    """
    query = "SELECT * FROM declarations WHERE 1=1"
    params = []

    if name_pattern:
        query += " AND name LIKE ?"
        params.append(f"%{name_pattern}%")

    if namespace:
        query += " AND namespace = ?"
        params.append(namespace)

    if file_path:
        query += " AND file_path LIKE ?"
        params.append(f"%{file_path}%")

    if decl_type:
        query += " AND decl_type = ?"
        params.append(decl_type)

    query += " LIMIT 1000"

    cursor = db.conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()

    if rows:
        columns = [description[0] for description in cursor.description]
        return pd.DataFrame([dict(row) for row in rows], columns=columns)
    else:
        return pd.DataFrame()


def get_all_declaration_names(db: LeanDeclarationDB) -> List[str]:
    """
    Get all unique declaration names for autocomplete.

    Args:
        db: LeanDeclarationDB instance

    Returns:
        List of declaration names
    """
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT DISTINCT name
        FROM declarations
        WHERE file_path IS NOT NULL
        ORDER BY name
    """)
    return [row[0] for row in cursor.fetchall()]


def get_declaration_types(db: LeanDeclarationDB) -> List[str]:
    """
    Get all unique declaration types.

    Args:
        db: LeanDeclarationDB instance

    Returns:
        List of declaration types
    """
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT DISTINCT decl_type
        FROM declarations
        WHERE decl_type IS NOT NULL
        ORDER BY decl_type
    """)
    return [row[0] for row in cursor.fetchall()]
