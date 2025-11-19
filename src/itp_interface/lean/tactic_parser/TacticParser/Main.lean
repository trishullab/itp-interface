/-
Main executable: read base64 from stdin, output JSON to stdout.
Runs in a loop to avoid restart overhead.

The process should be started from the project directory for project-specific parsing.
-/
import TacticParser.Base64
import TacticParser.Types
import TacticParser.SyntaxWalker
import TacticParser.LineParser
import Lean

open Lean
open TacticParser

/-- Result of parsing tactics -/
class FromStr (α : Type) where
fromStr : String → Option α

inductive ParseRequestType
  | parseTactics
  | parseTheorem
  | chkptTactics
  | breakChckpnt
deriving Inhabited, Repr, BEq

instance : ToString ParseRequestType where
  toString
    | .breakChckpnt => "break_chckpnt"
    | .chkptTactics => "chkpt_tactics"
    | .parseTactics => "parse_tactics"
    | .parseTheorem => "parse_theorem"

-- define the representation of ParseRequestType
def parse_request_names := [
  "break_chckpnt",
  "chkpt_tactics",
  "parse_tactics",
  "parse_theorem"
]

def parse_max_pad := parse_request_names.map String.length |>.foldl Nat.max 0

#eval parse_max_pad

instance : FromStr ParseRequestType where
  fromStr s :=
    match s with
    | "break_chckpnt" => some ParseRequestType.breakChckpnt
    | "chkpt_tactics" => some ParseRequestType.chkptTactics
    | "parse_tactics" => some ParseRequestType.parseTactics
    | "parse_theorem" => some ParseRequestType.parseTheorem
    | _ => none

structure UserParseRequest where
  requestType : ParseRequestType
  content : String -- Lean code
  deriving Inhabited, Repr, BEq

instance : FromStr UserParseRequest where
  fromStr s :=
    if s.length < parse_max_pad + 1 then
      none
    else
      let pref := s.take parse_max_pad
      let content := s.drop parse_max_pad
      match FromStr.fromStr pref with
      | some reqType => some { requestType := reqType, content := content }
      | none => none

#eval (FromStr.fromStr "parse_tactics" : Option ParseRequestType)

def some_lean_code : String := "parse_tactics
theorem test1 (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  apply And.intro
  exact hp
  exact hq
"
#eval! (FromStr.fromStr some_lean_code : Option UserParseRequest)

/-- Process a single request and output JSON -/
unsafe def processRequest (b64Input : String) (chkptState : Option CheckpointedParseResult := none) : IO  (Option CheckpointedParseResult) := do
  try
    -- Decode base64 to Lean code
    let parse_request_raw ← match Base64.decode b64Input with
      | .ok correct_parse_request => pure correct_parse_request
      | .error msg =>
        -- Output error as JSON
        let errorInfo := ErrorInfo.mk (s!"Base64 decode error: {msg}") { line := 0, column := 0 }
        let result : ParseResult := { trees := #[], errors := #[errorInfo] }
        IO.println (toJson result).compress
        return none

    let user_parse_request : Option UserParseRequest ← pure (FromStr.fromStr parse_request_raw)

    if user_parse_request.isNone then
      -- Output error as JSON
      let errorInfo := ErrorInfo.mk (s!"Invalid parse request format.") { line := 0, column := 0 }
      let result : ParseResult := { trees := #[], errors := #[errorInfo] }
      IO.println (toJson result).compress
      return none

    let parse_request := user_parse_request.get!

    let mut result : ParseResult := { trees := #[], errors := #[] }
    -- Initialize new checkpoint state to the current one
    let mut newchkptState : Option CheckpointedParseResult := chkptState
    let is_of_tactics_type :=
      parse_request.requestType == ParseRequestType.parseTactics ∨
      parse_request.requestType == ParseRequestType.chkptTactics ∨
      parse_request.requestType == ParseRequestType.breakChckpnt
    let is_checkpoint_request :=
      parse_request.requestType == ParseRequestType.chkptTactics
    let is_break_checkpoint_request :=
      parse_request.requestType == ParseRequestType.breakChckpnt
    if is_of_tactics_type then
      if is_break_checkpoint_request then
        -- First check if it is a breaking request, clear the last state
        newchkptState := none
      -- Parse tactics from Lean code
      let cmdState :=
        match newchkptState with
        | some chkpt => chkpt.chkptState
        | none => none
      let chkpointParseResult ← parseTactics parse_request.content none cmdState
      result := chkpointParseResult.parseResult
      --IO.println s!"Parsed tactics with {result.trees.size} trees and {repr result.errors} errors."
      if is_checkpoint_request then
        -- Only changes if the checkpoint is to be updated
        let line_num := chkpointParseResult.lineNum.getD 0
        let prev_line_num :=
          match newchkptState with
          | some chkpt => chkpt.lineNum.getD 0
          | none => 0
        -- Adjust line number based on previous checkpoint
        newchkptState := some {
          parseResult := chkpointParseResult.parseResult,
          lineNum := some (line_num + prev_line_num),
          chkptState := chkpointParseResult.chkptState
        }
      -- Additionally, adjust error positions based on previous checkpoint
      let prev_line_num :=
        match newchkptState with
        | some chkpt => chkpt.lineNum.getD 0
        | none => 0
      if prev_line_num > 0 then
        -- Adjust error line numbers
        let adjusted_errors := result.errors.map (fun err =>
          { err with
            position := {
              line := err.position.line + prev_line_num,
              column := err.position.column
            }
          })
        -- Adjust tree line numbers
        let adjusted_trees := result.trees.map (fun tree =>
          let rec adjust_tree (node : InfoNodeStruct) : InfoNodeStruct :=
          {
            node with
            startPos := {
              line := node.startPos.line + prev_line_num,
              column := node.startPos.column
            },
            endPos := {
              line := node.endPos.line + prev_line_num,
              column := node.endPos.column
            },
            children := node.children.map adjust_tree
          }
          adjust_tree tree
        )
        result := { trees := adjusted_trees, errors := adjusted_errors }
    else
      -- Unsupported request type
      let temp_result ← parseDecls parse_request.content
      let mut tree_list : Array InfoNodeStruct := #[]
      for decl in temp_result do
        let info_tree ← pure (InfoNodeStruct.mk decl.declType decl.name decl.docString decl.text decl.startPos decl.endPos decl.namespc #[])
        tree_list := tree_list.push info_tree
      result := { trees := tree_list, errors := #[] }

    -- Output result as JSON
    IO.println (toJson result).compress
    return newchkptState
  catch e =>
    -- Output error as JSON
    let errorInfo := ErrorInfo.mk (s!"Unexpected error: {e}") { line := 0, column := 0 }
    let result : ParseResult := { trees := #[], errors := #[errorInfo] }
    IO.println (toJson result).compress
    return none

/-- Loop to process requests -/
unsafe def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) (chkptState : Option CheckpointedParseResult := none) : IO Unit := do
  -- Read input from stdin (base64)
  let line ← stdin.getLine
  let line := line.trim

  -- Exit on empty line or "exit" command
  if line.isEmpty || line = "exit" then
    return

  -- Process the request
  let mut newchkptState ← processRequest line chkptState

  -- Flush output to ensure Python can read it
  stdout.flush

  -- Continue loop
  loop stdin stdout newchkptState

/-- Start processing -/
unsafe def main (args : List String) : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  loop stdin stdout none
