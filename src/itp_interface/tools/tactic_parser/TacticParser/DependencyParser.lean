import Lean
import Lean.Data.Json
import Lean.Elab.Frontend
import TacticParser.Types
import TacticParser.LineParser

namespace TacticParser

open Lean
open Lean.Parser
open Lean.Elab

/-- Extract module name from import syntax -/
partial def extractModuleName (stx : Syntax) : String :=
  match stx with
  | Syntax.ident _ _ name _ => name.toString
  | Syntax.node _ _ args =>
    -- Search through arguments to find identifiers and combine them
    let parts := args.filterMap fun arg =>
      let name := extractModuleName arg
      if name != "" then some name else none
    String.intercalate "." parts.toList
  | _ => ""

/-- Convert file path to module name -/
def filePathToModuleName (filepath : System.FilePath) : String :=
  let pathStr := filepath.toString
  -- Remove .lean extension
  let withoutExt := if pathStr.endsWith ".lean" then
    pathStr.dropRight 5
  else
    pathStr
  -- Replace path separators with dots
  let modulePath := withoutExt.replace "/" "."
  -- Remove leading ./ if present
  if modulePath.startsWith ".." then
    modulePath.drop 2
  else if modulePath.startsWith "." then
    modulePath.drop 1
  else
    modulePath

/-- Split a fully qualified name into namespace and local name -/
def splitNamespace (fullName : Name) : Option String × String :=
  let str := fullName.toString
  match str.splitOn "." with
  | [] => (none, str)
  | [single] => (none, single)
  | parts =>
    let ns := String.intercalate "." (parts.dropLast)
    let localPart := parts.getLast!
    (some ns, localPart)

/-- Get all constants used in an expression -/
def getConstantsFromExpr (e : Expr) : NameSet :=
  e.foldConsts {} fun c s => s.insert c

/-- Recursively collect all identifier names from a syntax tree (fallback for non-elaborated syntax) -/
partial def collectIdentifiers (stx : Syntax) : List Name :=
  match stx with
  | Syntax.ident _ _ name _ => [name]
  | Syntax.node _ _ args =>
    args.toList.flatMap collectIdentifiers
  | _ => []

/-- Resolve a constant name to its source information using the environment -/
def resolveConstant (env : Environment) (constName : Name) (moduleMap : Std.HashMap String String := {}) : Option DeclarationDependency := do
  -- Try to find the constant in the environment
  let constInfo? := env.find? constName
  match constInfo? with
  | none => none
  | some _ =>
    -- Get the module index where this constant is defined
    let moduleIdx? := env.getModuleIdx? constName
    let moduleName? := moduleIdx?.bind fun idx =>
      if h : idx.toNat < env.header.moduleNames.size then
        some env.header.moduleNames[idx.toNat]
      else
        none
    let moduleStr := moduleName?.map Name.toString

    -- Split into namespace and local name
    let (namespc, localPart) := splitNamespace constName

    -- Try to determine file path from module name
    let filePath? := match moduleStr with
      | some modName =>
        -- First check if it's in our import map (project-local imports)
        match moduleMap.get? modName with
        | some path => some path
        | none =>
          -- Otherwise use standard conversion for external libraries
          let path := modName.replace "." "/"
          some s!"{path}.lean"
      | none =>
        -- No module name from getModuleIdx, try to infer from namespace
        -- Since we can't definitively match, just return none for now
        none

    -- Also update module name if we inferred a file path but don't have module name yet
    let moduleStr := match (moduleStr, filePath?) with
      | (none, some fp) =>
        -- Try to reverse lookup module name from file path
        let entries := moduleMap.toList
        entries.find? (fun (modName, path) => path == fp) |>.map (·.1)
      | (some m, _) => some m
      | (none, none) =>
        -- No module info at all - only apply fallback heuristic if namespace
        -- doesn't look like stdlib (Lean, Init, Std, Nat, List, etc.)
        match namespc with
        | some ns =>
          let isStdlib := ns.startsWith "Lean" || ns.startsWith "Init" ||
                         ns.startsWith "Std" || ns == "Nat" || ns == "List" ||
                         ns == "Eq" || ns == "And" || ns == "Or" || ns == "String"
          if isStdlib then
            none
          else
            -- Find first non-Lean/non-Init import as best guess
            moduleMap.toList.find? (fun (modName, _) =>
              !modName.startsWith "Lean" && !modName.startsWith "Init"
            ) |>.map (·.1)
        | none => none

    -- If we have a module name but no file path, get it from the map
    let filePath? := match (filePath?, moduleStr) with
      | (none, some modName) => moduleMap.get? modName
      | (some fp, _) => some fp
      | _ => none

    some {
      name := constName.toString
      namespc := namespc
      localName := localPart
      filePath := filePath?
      moduleName := moduleStr
    }

/-- Extract dependencies from a declaration's syntax -/
def extractDependenciesFromSyntax (env : Environment) (stx : Syntax) : Array DeclarationDependency × Array String :=
  -- Collect all identifiers from the syntax
  let identifiers := collectIdentifiers stx

  -- Remove duplicates by converting to NameSet then back
  let uniqueNames := identifiers.foldl (fun set n => set.insert n) ({} : Lean.NameSet)

  -- Resolve each name and partition into resolved and unresolved
  uniqueNames.toArray.foldl (fun (deps, unres) name =>
    match resolveConstant env name with
    | some dep => (deps.push dep, unres)
    | none => (deps, unres.push name.toString)
  ) (#[], #[])

/-- Parse the header recursively to find all import commands -/
partial def findImports (stx : Syntax) (content: String) : IO (Array ImportInfo) → IO (Array ImportInfo)
  | accIO => do
    let acc ← accIO
    match stx with
    | Syntax.node _ kind args =>
      if kind == `Lean.Parser.Module.import then
        -- Found an import
        match stx.getRange? with
        | some range =>
          let moduleName := extractModuleName stx
          let text := content.extract range.start range.stop
          let info : ImportInfo := {
            moduleName := moduleName
            startPos := range.start.byteIdx
            endPos := range.stop.byteIdx
            text := text
          }
          return acc.push info
        | none => return acc
      else
        -- Recursively search children
        args.foldlM (fun acc child => findImports child content (pure acc)) acc
    | _ => return acc

/-- Parse imports and namespaces from a Lean 4 file -/
def parseImports (filepath : System.FilePath) : IO DependencyInfo := do
  let content ← IO.FS.readFile filepath
  let inputCtx := mkInputContext content filepath.toString

  -- Parse header which contains imports
  let (headerStx, parserState, _) ← parseHeader inputCtx

  let mut imports : Array ImportInfo := #[]
  let mut namespaces : Array NamespaceInfo := #[]

  -- Extract the underlying syntax from TSyntax
  let headerSyn : Syntax := headerStx



  imports ← findImports headerSyn content (pure imports)

  -- Now parse the rest of the file to find namespace declarations
  let env ← Lean.importModules #[] {} 0
  let opts := {}
  let pmctx : ParserModuleContext := {
    env := env
    options := opts
  }

  let mut pstate := parserState
  let mut done := false

  while !done do
    let startPos := pstate.pos
    let (stx, pstate', _msgs) := parseCommand inputCtx pmctx pstate {}
    pstate := pstate'

    -- Check if this is a namespace declaration
    if stx.getKind == `Lean.Parser.Command.namespace then
      match stx.getRange? with
      | some range =>
        let namespaceName := extractModuleName stx
        let text := content.extract range.start range.stop
        let info : NamespaceInfo := {
          name := namespaceName
          startPos := range.start.byteIdx
          endPos := range.stop.byteIdx
          text := text
        }
        namespaces := namespaces.push info
      | none => pure ()

    -- Check if we made progress or reached end
    if pstate.pos == startPos || inputCtx.input.atEnd pstate.pos then
      done := true

  -- Derive module name from file path
  let moduleName := filePathToModuleName filepath

  let result : DependencyInfo := {
    filePath := filepath.toString
    moduleName := moduleName
    imports := imports
    namespaces := namespaces
  }
  return result

/-- Extract constants from a ConstantInfo (type + value) -/
def extractConstantsFromConstInfo (cinfo : ConstantInfo) : NameSet :=
  let fromType := getConstantsFromExpr cinfo.type
  match cinfo.value? with
  | some val =>
    let fromVal := getConstantsFromExpr val
    -- Merge the two sets by converting to arrays and combining
    let combined := fromType.toArray ++ fromVal.toArray
    combined.foldl (fun s n => s.insert n) {}
  | none => fromType

/-- Analyze all declarations in a file and extract their dependencies -/
unsafe def analyzeFileDependencies (filepath : System.FilePath) : IO FileDependencyAnalysis := do
  -- Read file content
  let content ← IO.FS.readFile filepath

  -- Parse basic file structure (imports and namespaces)
  let depInfo ← parseImports filepath

  -- Parse all declarations using LineParser
  let decls ← parseFile filepath

  -- Load environment with all imports and elaborate the file
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution

  let inputCtx := Parser.mkInputContext content filepath.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let (env, _) ← processHeader header {} messages inputCtx

  -- Elaborate the entire file to get all declarations in the environment
  let commandState := Command.mkState env messages {}
  let finalState ← IO.processCommands inputCtx parserState commandState <&> Frontend.State.commandState
  let elaboratedEnv := finalState.env

  -- Build a map of module names to file paths from imports
  let mut moduleToFilePath : Std.HashMap String String := {}
  for imp in depInfo.imports do
    let filePath := imp.moduleName.replace "." "/" ++ ".lean"
    moduleToFilePath := moduleToFilePath.insert imp.moduleName filePath

  -- Analyze each declaration
  let mut declsWithDeps : Array DeclWithDependencies := #[]
  for declInfo in decls do
    -- Construct the fully qualified name for this declaration
    let declName := match declInfo.namespc with
      | some ns =>
        -- Build the name from namespace parts
        let namespaceParts := ns.splitOn "."
        let baseName := namespaceParts.foldl (fun n part => Name.mkStr n part) Name.anonymous
        Name.mkStr baseName declInfo.name
      | none => Name.mkStr Name.anonymous declInfo.name

    -- Try to find this declaration in the elaborated environment
    match elaboratedEnv.find? declName with
    | some constInfo =>
      -- Extract all constants from the type and value
      let allConstants := extractConstantsFromConstInfo constInfo

      -- Remove the declaration itself from its dependencies
      let allConstants := allConstants.erase declName

      -- Resolve each constant
      let mut dependencies : Array DeclarationDependency := #[]
      let mut unresolved : Array String := #[]

      for constName in allConstants.toArray do
        -- Check if constant exists in pre-elaboration env (means it's imported)
        match env.find? constName with
        | some _ =>
          -- It's from an import, use pre-elaboration env for module info
          match resolveConstant env constName moduleToFilePath with
          | some dep => dependencies := dependencies.push dep
          | none => unresolved := unresolved.push constName.toString
        | none =>
          -- It's a local declaration from this file, use elaborated env
          match resolveConstant elaboratedEnv constName moduleToFilePath with
          | some dep => dependencies := dependencies.push dep
          | none => unresolved := unresolved.push constName.toString

      let declWithDeps : DeclWithDependencies := {
        declInfo := declInfo
        dependencies := dependencies
        unresolvedNames := unresolved
      }
      declsWithDeps := declsWithDeps.push declWithDeps

    | none =>
      -- Declaration not found in elaborated environment (might be namespace, end, etc.)
      -- Fall back to syntax-based extraction
      let declInputCtx := Parser.mkInputContext declInfo.text "<decl>"
      let (_, declParserState, _) ← Parser.parseHeader declInputCtx
      let pmctx : ParserModuleContext := { env := env, options := {} }
      let (declStx, _, _) := Parser.parseCommand declInputCtx pmctx declParserState {}
      let (dependencies, unresolvedNames) := extractDependenciesFromSyntax env declStx

      let declWithDeps : DeclWithDependencies := {
        declInfo := declInfo
        dependencies := dependencies
        unresolvedNames := unresolvedNames
      }
      declsWithDeps := declsWithDeps.push declWithDeps

  let result : FileDependencyAnalysis := {
    filePath := filepath.toString
    moduleName := depInfo.moduleName
    imports := depInfo.imports
    declarations := declsWithDeps
  }
  return result

/-- Export dependency info to JSON file -/
def exportDependenciesToJson (info : DependencyInfo) (outputPath : System.FilePath) : IO Unit := do
  let json := toJson info
  let jsonStr := json.compress
  IO.FS.writeFile outputPath jsonStr
  IO.println s!"Exported dependencies to {outputPath}"

/-- Print import info -/
def printImportInfo (info : ImportInfo) : IO Unit := do
  IO.println s!"  - {info.moduleName}"
  IO.println s!"    Position: {info.startPos} - {info.endPos}"
  IO.println s!"    Text: {info.text}"

/-- Print namespace info -/
def printNamespaceInfo (info : NamespaceInfo) : IO Unit := do
  IO.println s!"  - {info.name}"
  IO.println s!"    Position: {info.startPos} - {info.endPos}"
  IO.println s!"    Text: {info.text}"

/-- Parse and print dependencies -/
def parseAndPrintDependencies (filepath : System.FilePath) : IO Unit := do
  IO.println s!"Analyzing dependencies for: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))

  let depInfo ← parseImports filepath

  IO.println s!"Module: {depInfo.moduleName}"
  IO.println ""

  if depInfo.imports.isEmpty then
    IO.println "No imports found."
  else
    IO.println s!"Imports ({depInfo.imports.size}):"
    for imp in depInfo.imports do
      printImportInfo imp

  IO.println ""

  if depInfo.namespaces.isEmpty then
    IO.println "No namespaces found."
  else
    IO.println s!"Namespaces ({depInfo.namespaces.size}):"
    for ns in depInfo.namespaces do
      printNamespaceInfo ns

/-- Parse and export dependencies -/
def parseDependenciesAndExport (filepath : System.FilePath) (jsonOutput : Option System.FilePath := none) : IO Unit := do
  let depInfo ← parseImports filepath

  -- Print to console
  IO.println s!"Analyzing dependencies for: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))

  IO.println s!"Module: {depInfo.moduleName}"
  IO.println ""

  if depInfo.imports.isEmpty then
    IO.println "No imports found."
  else
    IO.println s!"Imports ({depInfo.imports.size}):"
    for imp in depInfo.imports do
      printImportInfo imp

  IO.println ""

  if depInfo.namespaces.isEmpty then
    IO.println "No namespaces found."
  else
    IO.println s!"Namespaces ({depInfo.namespaces.size}):"
    for ns in depInfo.namespaces do
      printNamespaceInfo ns

  -- Export to JSON if output path provided
  match jsonOutput with
  | some outPath => exportDependenciesToJson depInfo outPath
  | none => pure ()

/-- Export FileDependencyAnalysis to JSON file -/
def exportFileDependencyAnalysisToJson (analysis : FileDependencyAnalysis) (outputPath : System.FilePath) : IO Unit := do
  let json := toJson analysis
  let jsonStr := json.compress
  IO.FS.writeFile outputPath jsonStr
  IO.println s!"Exported file dependency analysis to {outputPath}"

/-- Print declaration dependencies -/
def printDeclDependencies (decl : DeclWithDependencies) : IO Unit := do
  IO.println s!"[{decl.declInfo.declType}] {decl.declInfo.name}"
  if !decl.dependencies.isEmpty then
    IO.println s!"  Dependencies ({decl.dependencies.size}):"
    for dep in decl.dependencies do
      let modInfo := match dep.moduleName with
        | some m => s!" (from {m})"
        | none => ""
      IO.println s!"    - {dep.name}{modInfo}"
  if !decl.unresolvedNames.isEmpty then
    IO.println s!"  Unresolved ({decl.unresolvedNames.size}): {String.intercalate ", " decl.unresolvedNames.toList}"
  IO.println ""

/-- Analyze and print file dependencies with per-declaration tracking -/
unsafe def analyzeAndPrintFileDependencies (filepath : System.FilePath) : IO Unit := do
  IO.println s!"Analyzing file dependencies: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))

  let analysis ← analyzeFileDependencies filepath

  IO.println s!"Module: {analysis.moduleName}"
  IO.println s!"File: {analysis.filePath}"
  IO.println ""

  if !analysis.imports.isEmpty then
    IO.println s!"Imports ({analysis.imports.size}):"
    for imp in analysis.imports do
      IO.println s!"  - {imp.moduleName}"
    IO.println ""

  IO.println s!"Declarations with Dependencies ({analysis.declarations.size}):"
  IO.println ""
  for decl in analysis.declarations do
    printDeclDependencies decl

/-- Analyze and export file dependencies -/
unsafe def analyzeAndExportFileDependencies (filepath : System.FilePath) (jsonOutput : Option System.FilePath := none) : IO Unit := do
  let analysis ← analyzeFileDependencies filepath

  -- Print to console
  IO.println s!"Analyzing file dependencies: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))
  IO.println s!"Module: {analysis.moduleName}"
  IO.println s!"Declarations analyzed: {analysis.declarations.size}"

  -- Export to JSON if output path provided
  match jsonOutput with
  | some outPath => exportFileDependencyAnalysisToJson analysis outPath
  | none => pure ()

end TacticParser
