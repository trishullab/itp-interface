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
deriving Inhabited, Repr, BEq

instance : ToString ParseRequestType where
  toString
    | .parseTactics => "parse_tactics"
    | .parseTheorem => "parse_theorem"

-- define the representation of ParseRequestType
def parse_request_names := [
  "parse_tactics",
  "parse_theorem"
]

def parse_max_pad := parse_request_names.map String.length |>.foldl Nat.max 0

#eval parse_max_pad

instance : FromStr ParseRequestType where
  fromStr s :=
    match s with
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
      | some reqType => match reqType with
        | ParseRequestType.parseTactics =>
          some { requestType := reqType, content := content }
        | ParseRequestType.parseTheorem =>
          some { requestType := reqType, content := content }
      | none => none

#eval (FromStr.fromStr "parse_tactics" : Option ParseRequestType)

def some_lean_code : String := "parse_tactics
theorem test1 (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  apply And.intro
  exact hp
  exact hq
"
#eval (FromStr.fromStr some_lean_code : Option UserParseRequest)

/-- Process a single request and output JSON -/
unsafe def processRequest (b64Input : String) : IO Unit := do
  try
    -- Decode base64 to Lean code
    let parse_request_raw ← match Base64.decode b64Input with
      | .ok correct_parse_request => pure correct_parse_request
      | .error msg =>
        -- Output error as JSON
        let errorInfo := ErrorInfo.mk (s!"Base64 decode error: {msg}") { line := 0, column := 0 }
        let result : ParseResult := { trees := #[], errors := #[errorInfo] }
        IO.println (toJson result).compress
        return

    let user_parse_request : Option UserParseRequest ← pure (FromStr.fromStr parse_request_raw)

    if user_parse_request.isNone then
      -- Output error as JSON
      let errorInfo := ErrorInfo.mk (s!"Invalid parse request format.") { line := 0, column := 0 }
      let result : ParseResult := { trees := #[], errors := #[errorInfo] }
      IO.println (toJson result).compress
      return

    let parse_request := user_parse_request.get!

    let mut result : ParseResult := { trees := #[], errors := #[] }
    if parse_request.requestType == ParseRequestType.parseTactics then
      -- Parse tactics from Lean code
      result ← parseTactics parse_request.content
    else
      -- Unsupported request type
      let temp_result ← parseDecls parse_request.content
      let mut tree_list : Array (Option InfoTreeNode) := #[]
      for decl in temp_result do
        let start_pos ← pure (get_position_from_char_pos parse_request.content decl.startPos)
        let end_pos ← pure (get_position_from_char_pos parse_request.content decl.endPos)
        let info_tree: InfoTreeNode ← pure (InfoTreeNode.leanInfo decl.declType decl.name decl.docString decl.text start_pos end_pos #[])
        tree_list := tree_list.push (some info_tree)
      result := { trees := tree_list, errors := #[] }

    -- Output result as JSON
    IO.println (toJson result).compress

  catch e =>
    -- Output error as JSON
    let errorInfo := ErrorInfo.mk (s!"Unexpected error: {e}") { line := 0, column := 0 }
    let result : ParseResult := { trees := #[], errors := #[errorInfo] }
    IO.println (toJson result).compress

/-- Loop to process requests -/
unsafe def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) : IO Unit := do
  -- Read input from stdin (base64)
  let line ← stdin.getLine
  let line := line.trim

  -- Exit on empty line or "exit" command
  if line.isEmpty || line = "exit" then
    return

  -- Process the request
  processRequest line

  -- Flush output to ensure Python can read it
  stdout.flush

  -- Continue loop
  loop stdin stdout

/-- Start processing -/
unsafe def main (args : List String) : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  loop stdin stdout
