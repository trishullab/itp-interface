#!/usr/bin/env python3

import os
import sys
dir_name = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.abspath(dir_name)
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import uuid
import json
from pathlib import Path
from filelock import FileLock
from itp_interface.lean.simple_lean4_sync_executor import SimpleLean4SyncExecutor
from itp_interface.lean.parsing_helpers import LeanDeclType, preprocess_declarations
from itp_interface.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType
from itp_interface.tools.training_data_format import MergableCollection, TrainingDataMetadataFormat, ExtractionDataCollection
from itp_interface.tools.training_data import TrainingData, DataLayoutFormat
from itp_interface.lean.tactic_parser import FileDependencyAnalysis, DeclWithDependencies
from itp_interface.tools.simple_sqlite import LeanDeclarationDB

class Local4DataExtractionTransform(GenericTrainingDataGenerationTransform):
    def __init__(self,
                depth = None,
                max_search_results = None,
                buffer_size : int = 10000,
                logger = None,
                max_parallelism : int = 4,
                db_path : typing.Optional[str] = None,
                enable_file_export: bool = False,
                enable_dependency_extraction: bool = False):
        super().__init__(TrainingDataGenerationType.LOCAL, buffer_size, logger)
        self.depth = depth
        self.max_search_results = max_search_results
        self.max_parallelism = max_parallelism
        self.db_path = db_path  # Store path, don't create connection yet (for Ray actors)
        self.enable_file_export = enable_file_export
        self.enable_dependency_extraction = enable_dependency_extraction
        # Everything except UNKNOWN
        self.supported_declaration_types = [ str(dt) for dt in LeanDeclType if dt != LeanDeclType.UNKNOWN]
        if self.db_path is not None:
            temp_db_path = Path(self.db_path)
            cache_path = temp_db_path.parent / "cache"
            os.makedirs(str(cache_path), exist_ok=True)
            mapping_path = cache_path / f"{temp_db_path.stem}_declaration_mapping.json"
            self.cache_path = str(cache_path)
            self.mapping_path = str(mapping_path)
            self._lock_path = os.path.join(self.cache_path, "lean_declaration_mapping.lock")
        else:
            self.cache_path = None
            self.mapping_path = None
            self._lock_path = None

    def get_meta_object(self) -> TrainingDataMetadataFormat:
        return TrainingDataMetadataFormat(
            training_data_buffer_size=self.buffer_size,
            data_filename_prefix="extraction_data_",
            lemma_ref_filename_prefix="extraction_lemma_refs_")

    def get_data_collection_object(self) -> MergableCollection:
        return ExtractionDataCollection()
    
    def load_meta_from_file(self, file_path) -> MergableCollection:
        return TrainingDataMetadataFormat.load_from_file(file_path)
    
    def load_data_from_file(self, file_path) -> MergableCollection:
        return ExtractionDataCollection.load_from_file(file_path, self.logger)

    def _remove_lake_package_prefix(self, module_name: str) -> str:
        if module_name.startswith("lake.packages.") or module_name.startswith(".lake.packages."):
            parts = module_name.split('.')
            if len(parts) > 3:
                module_name = '.'.join(parts[3:])
            else:
                module_name = ''
        return module_name
    
    def _remove_lake_file_prefix(self, path: Path) -> str:
        parts = path.parts
        if ".lake" in parts:
            lake_index = parts.index(".lake")
            new_parts = parts[lake_index + 3:]
            new_path = Path(*new_parts)
            return str(new_path)
        return str(path)
    
    def _check_if_lean_core_file(self, file_path: Path) -> bool:
        parts = file_path.parts
        # If `leanprover--lean4---v4.24.0/src/lean/` is in the path, it's a core file
        for i in range(len(parts) - 4):
            if (parts[i].startswith("leanprover--lean4---v") and
                parts[i+1] == "src" and
                parts[i+2] == "lean"):
                return True
        return False
    
    def _check_if_lean_core_module(self, module_name: str) -> bool:
        parts = module_name.split('.')
        # leanprover--lean4---v4.24.0.src.lean
        idx = None
        for i in range(len(parts) - 2):
            if parts[i].startswith("leanprover--lean4---v"):
                idx = i
        if idx is not None:
            for i in range(idx, len(parts)):
                if parts[i] == "src" and i + 1 < len(parts) and parts[i + 1] == "lean":
                    return True
        return False

    def _strip_core_file_path(self, file_path: Path) -> str:
        # Take only the parts after `leanprover--lean4---v4.24.0/src/lean/`
        parts = file_path.parts
        for i in range(len(parts) - 4):
            if (parts[i].startswith("leanprover--lean4---v") and
                parts[i+1] == "src" and
                parts[i+2] == "lean"):
                new_parts = parts[i+3:]
                new_path = Path(*new_parts)
                return str(new_path)
        return str(file_path)

    def _strip_core_module_name(self, module_name: str) -> str:
        parts = module_name.split('.')
        # leanprover--lean4---v4.24.0.src.lean
        idx: int | None = None
        for i in range(len(parts) - 2):
            if parts[i].startswith("leanprover--lean4---v"):
                idx = i
        if idx is not None:
            for i in range(idx, len(parts)):
                if parts[i] == "src" and i + 1 < len(parts) and parts[i + 1] == "lean":
                    new_parts = parts[i + 2:]
                    return '.'.join(new_parts)
        return module_name

    def _get_file_path(self, project_path: str, file_path: str) -> str:
        fp = Path(file_path)
        pp = Path(project_path)
        fp_abs = fp.resolve()
        pp_abs = pp.resolve()
        if self._check_if_lean_core_file(fp_abs):
            stripped_path = self._strip_core_file_path(fp_abs)
            return stripped_path
        relative_path = fp_abs.relative_to(pp_abs)
        rel_file_path = self._remove_lake_file_prefix(relative_path)
        return str(rel_file_path)
    
    def _get_module_name(self, project_path: str, module_name: str) -> str:
        if self._check_if_lean_core_module(module_name):
            stripped_module = self._strip_core_module_name(module_name)
            return stripped_module
        pp = Path(project_path)
        pp_abs = pp.resolve()
        pp_module = str(pp_abs).replace('/', '.')
        pp_module = pp_module.lstrip('.')
        # self.logger.info(f"Project module prefix: {pp_module}")
        if module_name.startswith(pp_module):
            module_name = module_name[len(pp_module):]
        module_name = module_name.lstrip('.')
        # self.logger.info(f"Module name after removing project prefix: {module_name}")
        module_name = self._remove_lake_package_prefix(module_name)
        return module_name

    def _remap_local_dependency(self,
        project_path: str,
        file_dep_analyses: typing.List[FileDependencyAnalysis]) -> None:
            # Map local dependencies
            all_decls : dict[str, DeclWithDependencies] = {}
            for fda in file_dep_analyses:
                for decl in fda.declarations:
                    name = decl.decl_info.name
                    if name is not None:
                        all_decls[name] = decl
            for fda in file_dep_analyses:
                fda_rel_path = self._get_file_path(project_path, fda.file_path)
                # Remove the pp_module prefix from the fda module name
                fda_module_name = self._get_module_name(project_path, fda.module_name)
                for decl in fda.declarations:
                    for decl_dep in decl.dependencies:
                        if decl_dep.name in all_decls:
                            local_decl = all_decls[decl_dep.name]
                            if decl_dep.file_path is None and \
                            local_decl.decl_info.decl_type in self.supported_declaration_types:
                                # We need to ensure that it is locally defined.
                                decl_dep.file_path = fda_rel_path
                                if decl_dep.module_name is None:
                                    decl_dep.module_name = fda_module_name
                                if decl_dep.namespace is None:
                                    decl_dep.namespace = decl.decl_info.namespace
    
    def _preprocess_declarations(self,
        project_path: str,
        file_dep_analyses: typing.List[FileDependencyAnalysis]) -> None:
        # Preprocess declarations to set file paths and module names
        preprocess_declarations(file_dep_analyses)

    def __call__(self,
        training_data: TrainingData,
        project_id : str,
        lean_executor: SimpleLean4SyncExecutor,
        print_coq_executor_callback: typing.Callable[[], SimpleLean4SyncExecutor],
        theorems: typing.List[str] = None,
        other_args: dict = {}) -> TrainingData:
        file_path = lean_executor.main_file
        project_path = project_id
        rel_file_path = self._get_file_path(project_path, file_path)
        file_namespace = rel_file_path.replace('/', '.')
        self.logger.info(f"=========================Processing {file_namespace}=========================")

        # Create database connection for this Ray actor (if db_path is provided)
        db = None
        if self.db_path:
            db = LeanDeclarationDB(self.db_path)
            self.logger.info(f"Connected to database: {self.db_path}")

        try:
            if isinstance(theorems, list) and len(theorems) == 1 and theorems[0] == "*":
                theorems = None
            else:
                theorems = set(theorems) if theorems is not None else None
            cnt = 0
            temp_dir = os.path.join(training_data.folder, "temp") if self.cache_path is None else self.cache_path
            temp_dir = temp_dir.rstrip('/')
            if self.cache_path is None:
                os.makedirs(temp_dir, exist_ok=True)
            json_output_path = f"{temp_dir}/{file_namespace.replace('.', '_')}.lean.deps.json"
            already_mapped = False
            if self._lock_path is not None:
                with FileLock(self._lock_path):
                    assert self.mapping_path is not None
                    if not os.path.exists(self.mapping_path):
                        with open(self.mapping_path, "w") as f:
                            json.dump({}, f)
                    with open(self.mapping_path, "r") as f:
                        mapping_data = json.load(f)
                    if json_output_path in mapping_data:
                        already_mapped = True
                    else:
                        mapping_data[json_output_path] = True
                        with open(self.mapping_path, "w") as f:
                            json.dump(mapping_data, f)
            if already_mapped and os.path.exists(json_output_path):
                # Read from existing file
                self.logger.info(f"Using existing dependency extraction file: {json_output_path}")
                with open(json_output_path, 'r') as f:
                    data = json.load(f)
                file_dep_analyses = [FileDependencyAnalysis.model_validate(data)]
            else:   
                file_dep_analyses = lean_executor.extract_all_theorems_and_definitions(json_output_path=json_output_path)
            self.logger.info(f"Extracted {len(file_dep_analyses)} FileDependencyAnalysis objects from {file_namespace}")
            self.logger.info(f"file_dep_analyses: {file_dep_analyses}")
            assert len(file_dep_analyses) == 1, "Expected exactly one FileDependencyAnalysis object"

            last_decl_id = None
            self._preprocess_declarations(project_path, file_dep_analyses)
            self._remap_local_dependency(project_path, file_dep_analyses)

            for fda in file_dep_analyses:
                fda_rel_path = self._get_file_path(project_path, fda.file_path)
                # Remove the pp_module prefix from the fda module name
                fda_module_name = self._get_module_name(project_path, fda.module_name)

                # Insert file and imports into database (if db is enabled)
                if db and fda.imports:
                    db.insert_file_imports(fda_rel_path, fda_module_name, fda.imports)
                    self.logger.info(f"Inserted file and {len(fda.imports)} imports for {fda_rel_path}")

                for decl in fda.declarations:
                    # Get or create decl_id from database (or generate new one if no DB)
                    if db:
                        if decl.decl_info.decl_type not in self.supported_declaration_types:
                            self.logger.info(f"Skipping declaration '{decl.decl_info.name}' of unsupported type '{decl.decl_info.decl_type}'")
                            continue
                        decl_id = db.process_declaration(
                            fda_file_path=fda_rel_path,
                            fda_module_name=fda_module_name,
                            decl=decl,
                            enable_dependency_extraction=self.enable_dependency_extraction
                        )
                        self.logger.info(f"Processed declaration '{decl.decl_info.name}' with ID: {decl_id}")
                    else:
                        # Fallback: generate unique ID without database
                        timestamp = str(int(uuid.uuid1().time_low))
                        random_id = str(uuid.uuid4())
                        decl_id = f"{timestamp}_{random_id}"
                    if self.enable_file_export:
                        new_fda = FileDependencyAnalysis(
                        file_path=str(fda_rel_path),
                        module_name=fda_module_name,
                        imports=fda.imports,
                        declarations=[])
                        line_info = decl.decl_info
                        if theorems is not None and line_info.name not in theorems:
                            continue
                        decl.decl_id = decl_id
                        new_fda.declarations.append(decl)
                        training_data.merge(new_fda)
                    cnt += 1
                    last_decl_id = decl_id

            if last_decl_id:
                training_data.meta.last_proof_id = last_decl_id
            self.logger.info(f"===============Finished processing {file_namespace}=====================")
            self.logger.info(f"Total declarations processed in this transform: {cnt}")
            return training_data
        finally:
            # Clean up database connection
            if db:
                db.close()
                self.logger.info("Closed database connection")


if __name__ == "__main__":
    import os
    import logging
    import time
    os.chdir(root_dir)
    # project_dir = 'data/test/lean4_proj/'
    project_dir = 'data/test/Mathlib'
    # file_name = 'data/test/lean4_proj/Lean4Proj/Basic.lean'
    # file_name = 'data/test/Mathlib/.lake/packages/mathlib/Mathlib/Algebra/Divisibility/Basic.lean'
    home_path = str(Path.home())
    file_name = f'{home_path}/.elan/toolchains/leanprover--lean4---v4.24.0/src/lean/Init/Prelude.lean'
    project_id = project_dir #.replace('/', '.')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_path = f".log/local_data_generation_transform/data/{time_str}"
    log_path = f".log/local_data_generation_transform/log/{time_str}"
    log_file = f"{log_path}/local_data_generation_transform-{time_str}.log"
    db_path = f"{log_path}/lean_declarations.db"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    def _print_lean_executor_callback():
        search_lean_exec = SimpleLean4SyncExecutor(main_file=file_name, project_root=project_dir)
        search_lean_exec.__enter__()
        return search_lean_exec
    # Create transform with database enabled
    transform = Local4DataExtractionTransform(0, buffer_size=1000, db_path=db_path)
    logger.info(f"Using database: {db_path}")
    training_data = TrainingData(
        output_path, 
        "training_metadata.json",
        training_meta=transform.get_meta_object(), 
        logger=logger,
        layout=DataLayoutFormat.DECLARATION_EXTRACTION)
    with SimpleLean4SyncExecutor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True) as coq_exec:
        transform(training_data, project_id, coq_exec, _print_lean_executor_callback, theorems=["*"])
    save_info = training_data.save()
    logger.info(f"Saved training data to {save_info}")