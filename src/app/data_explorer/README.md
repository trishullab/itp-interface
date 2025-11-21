# Lean Declaration Database Explorer

A Streamlit web application for exploring and analyzing Lean declaration databases created by the ITP Interface data extraction tool.

## Features

- **Custom SQL Queries**: Write and execute custom SQL queries with pre-built query templates
- **Declaration Search**: Search declarations by name, type, file path, and namespace
- **Dependency Explorer**: Visualize dependencies and dependents of declarations
- **Forest Analysis**: Analyze connected components in the dependency graph

## Installation

Install the required dependencies using pip with the `app` extra:

```bash
pip install -e ".[app]"
```

This will install:
- streamlit
- plotly
- networkx
- pandas

## Usage

1. **Generate a database** using the data extraction transform:
   ```python
   from itp_interface.tools.lean4_local_data_extraction_transform import Local4DataExtractionTransform

   transform = Local4DataExtractionTransform(
       buffer_size=1000,
       db_path="lean_declarations.db"
   )
   # ... run your extraction ...
   ```

2. **Launch the explorer**:
   ```bash
   cd src/app/data_explorer
   streamlit run lean_db_explorer.py
   ```

3. **Load your database**:
   - Enter the path to your `.db` file in the sidebar
   - Click "Load Database"
   - Explore using the tabs!

## App Tabs

### 🔍 Custom Query
Write and execute custom SQL queries against the database. Includes pre-built queries for common tasks:
- Show all files
- Show all declarations
- Count declarations by type
- Find most depended-on declarations
- And more...

### 🔎 Search
Search declarations with filters:
- Name pattern (supports partial matching)
- File path
- Declaration type (theorem, def, axiom, etc.)
- Namespace

### 🌲 Dependencies
Explore dependency relationships:
- **Show Dependencies**: What does this declaration depend on?
- **Show Dependents**: What depends on this declaration?
- **Configurable depth**: Control how deep to traverse
- **Dual view**: Table and interactive graph visualization

### 🌳 Forests
Analyze connected components in the dependency graph:
- **Find All Forests**: Discover all connected components
- **Statistics**: See forest sizes, roots, and leaves
- **Visualization**: View selected forests as graphs

## Database Schema

The app expects a SQLite database with the following tables:

- `files`: File metadata
- `imports`: Import relationships
- `declarations`: All declarations with full info
- `declaration_dependencies`: Dependency edges (from_decl_id → to_decl_id)

See `src/itp_interface/tools/simple_sqlite.py` for the complete schema.

## Tips

- **Large graphs**: Forest visualization is limited to 100 nodes for performance
- **Query results**: All query results can be downloaded as CSV
- **Unresolved declarations**: Declarations with `file_path IS NULL` are unresolved (from dependencies)

## Troubleshooting

**Database not loading?**
- Check the file path is correct
- Ensure the database was created with the correct schema

**Graph visualization slow?**
- Reduce the max depth for dependency exploration
- Use filters in Search to narrow results

**Import errors?**
- Ensure you've installed the app dependencies: `pip install -e ".[app]"`
- Run from the correct directory: `cd src/app/data_explorer`

## Development

File structure:
```
src/app/data_explorer/
├── lean_db_explorer.py    # Main Streamlit app
├── db_utils.py             # Database query utilities
├── graph_utils.py          # Graph analysis and visualization
└── README.md               # This file
```

To modify:
1. Edit the Python files
2. Streamlit will auto-reload on file changes
3. Refresh browser to see updates
