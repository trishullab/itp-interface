"""
Lean Declaration Database Explorer

A Streamlit web app for exploring and analyzing Lean declaration databases.
"""

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
from itp_interface.tools.simple_sqlite import LeanDeclarationDB
import db_utils
import graph_utils


# Page configuration
st.set_page_config(
    page_title="Lean Declaration DB Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_database(db_path: str) -> LeanDeclarationDB:
    """Load database and cache in session state."""
    try:
        db = LeanDeclarationDB(db_path)
        return db
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        return None


def display_statistics(db: LeanDeclarationDB):
    """Display database statistics in the sidebar."""
    stats = db.get_statistics()

    st.sidebar.markdown("### 📊 Database Statistics")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric("Files", stats.get('total_files', 0))
        st.metric("Declarations", stats.get('total_declarations', 0))

    with col2:
        st.metric("Dependencies", stats.get('total_dependencies', 0))
        st.metric("Imports", stats.get('total_imports', 0))

    st.sidebar.metric("Unresolved", stats.get('unresolved_declarations', 0))


def tab_custom_query(db: LeanDeclarationDB):
    """Custom Query tab interface."""
    st.header("🔍 Custom SQL Query")

    # Example queries dropdown
    st.markdown("**Pre-built Queries:**")
    common_queries = db_utils.get_common_queries()
    query_name = st.selectbox(
        "Select a query to load",
        [""] + list(common_queries.keys()),
        label_visibility="collapsed"
    )

    # Query text area - use query_name as part of key to force refresh
    if query_name:
        default_query = common_queries.get(query_name)
    else:
        default_query = "SELECT * FROM declarations LIMIT 10"

    query = st.text_area(
        "SQL Query",
        value=default_query,
        height=150,
        key=f"custom_query_{query_name}"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        execute_button = st.button("Execute Query", type="primary")

    if execute_button:
        try:
            with st.spinner("Executing query..."):
                df = db_utils.execute_query(db, query)

            if not df.empty:
                st.success(f"Query returned {len(df)} rows")
                st.dataframe(df, use_container_width=True, height=500)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("Query returned no results")

        except Exception as e:
            st.error(f"Query error: {e}")


def tab_search(db: LeanDeclarationDB):
    """Search & Browse tab interface."""
    st.header("🔎 Search Declarations")

    col1, col2 = st.columns(2)

    with col1:
        name_pattern = st.text_input("Name pattern", placeholder="e.g., add, mul, theorem")
        file_path = st.text_input("File path contains", placeholder="e.g., Mathlib/Data")

    with col2:
        decl_types = [""] + db_utils.get_declaration_types(db)
        decl_type = st.selectbox("Declaration type", decl_types)

        namespace = st.text_input("Namespace", placeholder="e.g., Nat, List")

    if st.button("Search", type="primary"):
        with st.spinner("Searching..."):
            df = db_utils.search_declarations(
                db,
                name_pattern=name_pattern if name_pattern else None,
                namespace=namespace if namespace else None,
                file_path=file_path if file_path else None,
                decl_type=decl_type if decl_type else None
            )

        if not df.empty:
            st.success(f"Found {len(df)} declarations")

            # Show results
            st.dataframe(df, use_container_width=True, height=500)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="search_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No declarations found matching the criteria")


def tab_dependencies(db: LeanDeclarationDB):
    """Dependency Explorer tab interface."""
    st.header("🌲 Dependency Explorer")

    # Declaration selector
    st.markdown("**Select a declaration:**")
    decl_name = st.text_input("Declaration name", placeholder="e.g., dvd_trans, Nat.add")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_deps = st.button("Show Dependencies", type="primary")
    with col2:
        show_dependents = st.button("Show Dependents", type="primary")
    with col3:
        max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=5)

    if show_deps or show_dependents:
        if not decl_name:
            st.warning("Please enter a declaration name")
            return

        # Find declaration by name
        search_df = db_utils.search_declarations(db, name_pattern=decl_name)

        if search_df.empty:
            st.error(f"Declaration '{decl_name}' not found")
            return

        if len(search_df) > 1:
            st.warning(f"Found {len(search_df)} declarations with this name. Using the first one.")
            st.dataframe(search_df[['name', 'namespace', 'file_path', 'decl_type']])

        decl_id = search_df.iloc[0]['decl_id']

        with st.spinner("Analyzing dependencies..."):
            if show_deps:
                decls, subgraph = graph_utils.get_dependencies_closure(db, decl_id, max_depth)
                title = f"Dependencies of {decl_name}"
            else:
                decls, subgraph = graph_utils.get_dependents_closure(db, decl_id, max_depth)
                title = f"Dependents of {decl_name}"

        if decls:
            st.success(f"Found {len(decls)} related declarations")

            # Show tabs for table and graph views
            tab1, tab2 = st.tabs(["📊 Table View", "📈 Graph View"])

            with tab1:
                df = pd.DataFrame(decls)
                st.dataframe(df[['name', 'namespace', 'decl_type', 'file_path', 'line']],
                             use_container_width=True, height=400)

            with tab2:
                fig = graph_utils.visualize_graph(subgraph, title)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dependencies found")


def tab_forests(db: LeanDeclarationDB):
    """Forest Analysis tab interface."""
    st.header("🌳 Forest Analysis")

    st.markdown("""
    A **forest** is a connected component in the dependency graph. Declarations in the same
    forest are connected through dependency relationships.
    """)

    # Initialize session state for forests
    if 'forests' not in st.session_state:
        st.session_state.forests = None
        st.session_state.forest_graph = None

    if st.button("Find All Forests", type="primary"):
        with st.spinner("Analyzing dependency graph..."):
            G = graph_utils.build_dependency_graph(db)
            forests = graph_utils.find_all_forests(G)
            # Store in session state
            st.session_state.forests = forests
            st.session_state.forest_graph = G

        st.success(f"Found {len(forests)} forests")

    # Use forests from session state
    forests = st.session_state.forests
    G = st.session_state.forest_graph

    if forests is not None:
        # Display statistics
        st.markdown("### Forest Statistics")

        forest_data = []
        for i, forest in enumerate(forests[:20]):  # Limit to top 20
            stats = graph_utils.get_forest_statistics(G, forest)
            forest_data.append({
                'Forest ID': i + 1,
                'Size': stats['size'],
                'Edges': stats['num_edges'],
                'Roots': stats['num_roots'],
                'Leaves': stats['num_leaves'],
                'Sample Roots': ', '.join(stats['roots'][:3])
            })

        df = pd.DataFrame(forest_data)
        st.dataframe(df, use_container_width=True)

        # Show declarations in selected forest
        st.markdown("### View Forest Declarations")
        forest_id = st.number_input(
            "Forest ID to view",
            min_value=1,
            max_value=len(forests),
            value=1
        )

        if st.button("Show Declarations in Forest"):
            forest = forests[forest_id - 1]
            st.info(f"Forest #{forest_id} contains {len(forest)} declarations")

            # Get all declarations in this forest
            forest_decls = []
            for decl_id in forest:
                decl = db.get_declaration_by_decl_id(decl_id)
                if decl:
                    # Truncate text and proof for display
                    text = decl.get('text', '')
                    proof = decl.get('proof', '')
                    forest_decls.append({
                        'decl_id': decl['decl_id'],
                        'name': decl['name'],
                        'namespace': decl.get('namespace', ''),
                        'decl_type': decl.get('decl_type', ''),
                        'text': text[:100] + '...' if text and len(text) > 100 else text,
                        'proof': proof[:100] + '...' if proof and len(proof) > 100 else proof,
                        'file_path': decl.get('file_path', ''),
                        'line': decl.get('line', '')
                    })

            forest_df = pd.DataFrame(forest_decls)
            st.dataframe(forest_df, use_container_width=True, height=500)

            # Download button
            csv = forest_df.to_csv(index=False)
            st.download_button(
                label="Download Forest Declarations",
                data=csv,
                file_name=f"forest_{forest_id}_declarations.csv",
                mime="text/csv"
            )
    else:
        st.info("Click 'Find All Forests' to analyze the dependency graph")


def main():
    """Main application."""
    st.title("📊 Lean Declaration Database Explorer")

    # Sidebar
    st.sidebar.title("Database Connection")

    # Database path input
    db_path = st.sidebar.text_input(
        "Database Path",
        value="lean_declarations.db",
        help="Path to the SQLite database file"
    )

    load_button = st.sidebar.button("Load Database", type="primary")

    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = None

    if load_button:
        if Path(db_path).exists():
            with st.spinner("Loading database..."):
                st.session_state.db = load_database(db_path)

            if st.session_state.db:
                st.sidebar.success("Database loaded successfully!")
        else:
            st.sidebar.error(f"Database file not found: {db_path}")

    # Show statistics if database is loaded
    if st.session_state.db:
        display_statistics(st.session_state.db)

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Custom Query",
            "🔎 Search",
            "🌲 Dependencies",
            "🌳 Forests"
        ])

        with tab1:
            tab_custom_query(st.session_state.db)

        with tab2:
            tab_search(st.session_state.db)

        with tab3:
            tab_dependencies(st.session_state.db)

        with tab4:
            tab_forests(st.session_state.db)

    else:
        st.info("👈 Please load a database to get started")

        st.markdown("""
        ### Getting Started

        1. Enter the path to your SQLite database file in the sidebar
        2. Click "Load Database"
        3. Explore your data using the tabs above

        ### Features

        - **Custom Query**: Write and execute custom SQL queries
        - **Search**: Search declarations by name, type, file, etc.
        - **Dependencies**: Explore dependency relationships
        - **Forests**: Analyze connected components in the dependency graph
        """)


if __name__ == "__main__":
    main()
