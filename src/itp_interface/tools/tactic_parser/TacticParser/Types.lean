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

/-- Information extracted from a declaration -/
structure DeclInfo where
  declType : DeclType
  name : String
  startPos : Nat
  endPos : Nat
  text : String
  docString : Option String  -- Extracted documentation comment
  namespc : Option String  -- Current namespace
  deriving Repr

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
  trees : Array (Option InfoTreeNode) := #[]
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
