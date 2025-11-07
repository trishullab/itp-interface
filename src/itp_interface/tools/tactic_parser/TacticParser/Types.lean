/-
Types for tactic information.
-/
import Lean
import Lean.Elab.Frontend

namespace TacticParser

open Lean
open Lean.Elab

/-- Represents different types of Lean declarations -/
inductive DeclType where
  | inductive
  | theorem
  | def
  | axiom
  | structure
  | class_decl
  | instance
  | other
  | example
  | lemma
  | unknown
  | tactic
  | namespace
  | end
  deriving Repr, BEq

instance : ToString DeclType where
  toString
    | .inductive => "inductive"
    | .theorem => "theorem"
    | .def => "def"
    | .axiom => "axiom"
    | .structure => "structure"
    | .class_decl => "class"
    | .instance => "instance"
    | .other => "other"
    | .example => "example"
    | .lemma => "lemma"  -- lemma is treated as theorem
    | .unknown => "unknown"
    | .tactic => "tactic"
    | .namespace => "namespace"
    | .end => "end"

/-- Position information for a tactic -/
structure Position where
  line : Nat
  column : Nat
  deriving Inhabited, Repr, BEq

instance : ToJson Position where
  toJson p := Json.mkObj [
    ("line", toJson p.line),
    ("column", toJson p.column)
  ]

instance : FromJson Position where
  fromJson? j := do
    let line ← j.getObjValAs? Nat "line"
    let column ← j.getObjValAs? Nat "column"
    return { line, column }

/-- Information extracted from a declaration -/
structure DeclInfo where
  declType : DeclType
  name : String
  startPos : Position
  endPos : Position
  text : String
  docString : Option String  -- Extracted documentation comment
  namespc : Option String  -- Current namespace
  deriving Repr

instance : ToJson DeclInfo where
  toJson d := Json.mkObj [
    ("decl_type", toJson (ToString.toString d.declType)),
    ("name", toJson d.name),
    ("line", d.startPos.line),
    ("column", d.startPos.column),
    ("end_line", d.endPos.line),
    ("end_column", d.endPos.column),
    ("text", toJson d.text),
    ("doc_string", toJson d.docString),
    ("namespace", toJson d.namespc)
  ]

/-- Information about an import statement -/
structure ImportInfo where
  moduleName : String
  startPos : Nat
  endPos : Nat
  text : String
  deriving Repr

instance : ToJson ImportInfo where
  toJson i := Json.mkObj [
    ("module_name", toJson i.moduleName),
    ("start_pos", toJson i.startPos),
    ("end_pos", toJson i.endPos),
    ("text", toJson i.text)
  ]

/-- Information about a namespace declaration -/
structure NamespaceInfo where
  name : String
  startPos : Nat
  endPos : Nat
  text : String
  deriving Repr

instance : ToJson NamespaceInfo where
  toJson n := Json.mkObj [
    ("name", toJson n.name),
    ("start_pos", toJson n.startPos),
    ("end_pos", toJson n.endPos),
    ("text", toJson n.text)
  ]

/-- Information about file dependencies and module structure -/
structure DependencyInfo where
  filePath : String
  moduleName : String  -- The module name derived from file path
  imports : Array ImportInfo
  namespaces : Array NamespaceInfo
  deriving Repr

instance : ToJson DependencyInfo where
  toJson d := Json.mkObj [
    ("file_path", toJson d.filePath),
    ("module_name", toJson d.moduleName),
    ("imports", toJson d.imports),
    ("namespaces", toJson d.namespaces)
  ]

/-- Information about a single dependency reference -/
structure DeclarationDependency where
  name : String              -- Fully qualified name (e.g., "Nat.add_zero")
  namespc : Option String    -- Namespace portion (e.g., "Nat")
  localName : String         -- Local name without namespace
  filePath : Option String   -- Source file if resolvable
  moduleName : Option String -- Module where defined
  deriving Repr

instance : ToJson DeclarationDependency where
  toJson d := Json.mkObj [
    ("name", toJson d.name),
    ("namespace", toJson d.namespc),
    ("local_name", toJson d.localName),
    ("file_path", toJson d.filePath),
    ("module_name", toJson d.moduleName)
  ]

/-- Declaration with its dependencies -/
structure DeclWithDependencies where
  declInfo : DeclInfo                       -- From LineParser
  dependencies : Array DeclarationDependency
  unresolvedNames : Array String            -- Names we couldn't resolve
  deriving Repr

instance : ToJson DeclWithDependencies where
  toJson d := Json.mkObj [
    ("decl_info", toJson d.declInfo),
    ("dependencies", toJson d.dependencies),
    ("unresolved_names", toJson d.unresolvedNames)
  ]

/-- Complete file dependency analysis with per-declaration tracking -/
structure FileDependencyAnalysis where
  filePath : String
  moduleName : String
  imports : Array ImportInfo
  declarations : Array DeclWithDependencies
  deriving Repr

instance : ToJson FileDependencyAnalysis where
  toJson f := Json.mkObj [
    ("file_path", toJson f.filePath),
    ("module_name", toJson f.moduleName),
    ("imports", toJson f.imports),
    ("declarations", toJson f.declarations)
  ]

/-- InfoTree node representation -/
inductive InfoTreeNode where
  | context : InfoTreeNode → InfoTreeNode
  | leanInfo
    (declType: DeclType)
    (name: Option String)
    (docString: Option String)
    (text: String)
    (startPos: Position)
    (endPos: Position)
    (namespc: Option String)
    (children: Array InfoTreeNode) : InfoTreeNode
  | other : Array InfoTreeNode → InfoTreeNode
  | hole : InfoTreeNode
  deriving Inhabited, Repr

partial def InfoTreeNode.toJson : InfoTreeNode → Json
  | .context child =>
    Json.mkObj [
      ("type", "context"),
      ("children", child.toJson)
    ]
  | leanInfo declType name docString text startPos endPos namespc children =>
    Json.mkObj [
      ("type", "leanInfo"),
      ("decl_type", ToString.toString declType),
      ("name", Lean.ToJson.toJson name),
      ("doc_string", Lean.ToJson.toJson docString),
      ("text", Lean.ToJson.toJson text),
      ("start_pos", Lean.ToJson.toJson startPos),
      ("end_pos", Lean.ToJson.toJson endPos),
      ("namespace", Lean.ToJson.toJson namespc),
      ("children", Json.arr (children.map InfoTreeNode.toJson))
    ]
  | .other children =>
    Json.mkObj [
      ("type", "other"),
      ("children", Json.arr (children.map InfoTreeNode.toJson))
    ]
  | .hole =>
    Json.mkObj [("type", "hole")]

instance : ToJson InfoTreeNode where
  toJson := InfoTreeNode.toJson

structure InfoNodeStruct where
  declType : DeclType
  name : Option String
  docString : Option String
  text : String
  startPos : Position
  endPos : Position
  namespc : Option String
  children : Array InfoNodeStruct
deriving Repr

def defaultInfoNodeStruct : InfoNodeStruct :=
  {
    declType := .unknown,
    name := none,
    docString := none,
    text := "",
    startPos := { line := 0, column := 0 },
    endPos := { line := 0, column := 0 },
    namespc := none,
    children := #[]
  }

instance : Inhabited InfoNodeStruct where
  default := defaultInfoNodeStruct

/-- toJson for InfoNodeStruct -/
partial def InfoNodeStruct.toJson (n: InfoNodeStruct) : Json :=
  Json.mkObj [
    ("decl_type", ToString.toString n.declType),
    ("name", Lean.ToJson.toJson n.name),
    ("doc_string", Lean.ToJson.toJson n.docString),
    ("text", Lean.ToJson.toJson n.text),
    ("start_pos", Lean.ToJson.toJson n.startPos),
    ("end_pos", Lean.ToJson.toJson n.endPos),
    ("namespace", Lean.ToJson.toJson n.namespc),
    ("children", Json.arr (n.children.map InfoNodeStruct.toJson))
  ]

instance : ToJson InfoNodeStruct where
  toJson := InfoNodeStruct.toJson

partial def getInfoNodeStruct (node : InfoTreeNode) : Option InfoNodeStruct :=
  match node with
  | .leanInfo declType name docString text startPos endPos namespc children =>
    let childStructs := children.map getInfoNodeStruct
    let filterSomes := childStructs.filterMap id
    some {
      declType,
      name,
      docString,
      text,
      startPos,
      endPos,
      namespc,
      children := filterSomes
    }
  | _ => none

structure ErrorInfo where
  message : String
  position : Position
  deriving Inhabited, Repr

instance : ToJson ErrorInfo where
  toJson e := Json.mkObj [
    ("message", toJson e.message),
    ("position", toJson e.position)
  ]

/-- Result of parsing tactics from Lean code -/
structure ParseResult where
  trees : Array InfoNodeStruct := #[]
  errors : Array ErrorInfo := #[]
  deriving Inhabited, Repr

instance : ToJson ParseResult where
  toJson r := Json.mkObj [
    ("trees", toJson r.trees),
    ("errors", toJson r.errors)
  ]

/-- Storing ParseResult with a checkpointed state -/
structure CheckpointedParseResult where
  parseResult : ParseResult
  chkptState : Option Command.State := none
  lineNum : Option Nat := none
  deriving Inhabited

/-- Custom Repr instance for CheckpointedParseResult -/
instance : Repr CheckpointedParseResult where
  reprPrec r _ := (repr r.parseResult)

def get_position_from_char_pos (content : String) (charPos : Nat) : Position :=
  let before := content.extract ⟨0⟩ ⟨charPos⟩
  let lines := before.splitOn "\n"
  let lineCount := lines.length
  if lineCount == 0 then
    { line := 0, column := 0 }
  else
    let lastLine := lines[lineCount - 1]!
    -- let byteLen := lastLine.endPos.byteIdx
    { line := lineCount, column := lastLine.length }

end TacticParser
