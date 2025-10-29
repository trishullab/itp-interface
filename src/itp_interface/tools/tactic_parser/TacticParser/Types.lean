/-
Types for tactic information.
-/
import Lean

namespace TacticParser

open Lean

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
  | tacticInfo : String → Position → Position → Array InfoTreeNode → InfoTreeNode
  | other : Array InfoTreeNode → InfoTreeNode
  | hole : InfoTreeNode
  deriving Inhabited, Repr

partial def InfoTreeNode.toJson : InfoTreeNode → Json
  | .context child =>
    Json.mkObj [
      ("type", "context"),
      ("children", child.toJson)
    ]
  | .tacticInfo text startPos endPos children =>
    Json.mkObj [
      ("type", "tacticInfo"),
      ("text", Lean.ToJson.toJson text),
      ("start_pos", Lean.ToJson.toJson startPos),
      ("end_pos", Lean.ToJson.toJson endPos),
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

/-- Result of parsing tactics from Lean code -/
structure ParseResult where
  trees : Array (Option InfoTreeNode) := #[]
  error : Option String := none
  deriving Inhabited, Repr

instance : ToJson ParseResult where
  toJson r := Json.mkObj [
    ("trees", toJson r.trees),
    ("error", toJson r.error)
  ]

end TacticParser
