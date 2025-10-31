/-
Syntax walker to extract tactics from Lean code with lightweight elaboration.
Uses InfoTrees (like REPL) which requires elaboration but NOT compilation!

Can work in two modes:
1. Standalone: Parse simple tactics without dependencies (minimal environment)
2. Project mode: Use a project's search path to enable mathlib/dependencies
-/
import Lean
import Lean.Elab.Frontend
import TacticParser.Types

open Lean Elab

namespace Lean.Elab.IO

/--
Wrapper for `IO.processCommands` that enables info states, and returns
* the new command state
* messages
* info trees
-/
def processCommandsWithInfoTrees
    (inputCtx : Parser.InputContext) (parserState : Parser.ModuleParserState)
    (commandState : Command.State) : IO (Command.State × Array Message × Array InfoTree) := do
  let commandState := { commandState with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState <&> Frontend.State.commandState
  pure (s, s.messages.toArray, s.infoState.trees.toArray)

/--
Process some text input, with or without an existing command state.
If there is no existing environment, we parse the input for headers (e.g. import statements),
and create a new environment.
Otherwise, we add to the existing environment.

Returns:
1. The header-only command state (only useful when cmdState? is none)
2. The resulting command state after processing the entire input
3. List of messages
4. List of info trees
-/
def processInput (input : String) (cmdState? : Option Command.State)
    (opts : Options := {}) (fileName : Option String := none) :
    IO (Command.State × Command.State × Array Message × Array InfoTree) := unsafe do
  Lean.initSearchPath (← Lean.findSysroot)
  enableInitializersExecution
  let fileName   := fileName.getD "<input>"
  let inputCtx   := Parser.mkInputContext input fileName

  match cmdState? with
  | none => do
    -- Split the processing into two phases to prevent self-reference in proofs in tactic mode
    let (header, parserState, messages) ← Parser.parseHeader inputCtx
    let (env, messages) ← processHeader header opts messages inputCtx
    let headerOnlyState := Command.mkState env messages opts
    let (cmdState, messages, trees) ← processCommandsWithInfoTrees inputCtx parserState headerOnlyState
    return (headerOnlyState, cmdState, messages, trees)

  | some cmdStateBefore => do
    let parserState : Parser.ModuleParserState := {}
    let (cmdStateAfter, messages, trees) ← processCommandsWithInfoTrees inputCtx parserState cmdStateBefore
    return (cmdStateBefore, cmdStateAfter, messages, trees)

end Lean.Elab.IO

namespace TacticParser

open Lean
open Lean.Elab
open Lean.Parser
open Lean.Syntax

/-- Convert a String.Pos to line and column numbers -/
def posToLineColumn (input : String) (pos : String.Pos) : Position :=
  let lines := input.extract 0 pos |>.splitOn "\n"
  let line := lines.length
  let column := (lines.getLast!).length
  { line, column }

/-- Extract the source text for a syntax node -/
def syntaxToString (stx : Syntax) : String :=
  stx.reprint.getD (toString stx)

/-- Pretty print InfoTree structure for debugging -/
partial def printInfoTree (input : String) (tree : InfoTree) (indent : Nat := 0) : IO Unit := do
  let spaces := String.pushn "" ' ' indent
  match tree with
  | .context _ t =>
    --IO.println s!"{spaces}Context"
    printInfoTree input t (indent)
  | .node info children =>
    match info with
    | .ofTacticInfo tacInfo =>
      -- Extract actual text from source using byte positions
      let startByte := tacInfo.stx.getPos?.getD 0
      let endByte := tacInfo.stx.getTailPos?.getD 0
      let actualText := input.extract startByte endByte |>.trim

      let startPos := posToLineColumn input startByte
      let endPos := posToLineColumn input endByte
      let preview := actualText
      IO.println s!"{spaces}TacticInfo: L{startPos.line}:C{startPos.column}-L{endPos.line}:C{endPos.column} | {preview.replace "\n" "\\n"}"
      for child in children do
        printInfoTree input child (indent + 2)
    | _ =>
      --IO.println s!"{spaces}Other"
      for child in children do
        printInfoTree input child (indent)
  | .hole _ =>
    IO.println s!"{spaces}Hole"

/-- Convert InfoTree to InfoTreeNode -/
partial def infoTreeToNode (input : String) (tree : InfoTree) : InfoTreeNode :=
  match tree with
  | .context _ t =>
    infoTreeToNode input t
  | .node info children =>
    let childNodes := (children.map (infoTreeToNode input)).toArray
    -- filter all children that are .hole and .other
    let filteredChildren := childNodes.filter fun
      | .hole => false
      | .other arr => arr.isEmpty
      | _ => true
    match info with
    | .ofTacticInfo tacInfo =>
      let text := tacInfo.stx.reprint.getD (toString tacInfo.stx) |>.trim
      let startPos := posToLineColumn input (tacInfo.stx.getPos?.getD 0)
      let endPos := posToLineColumn input (tacInfo.stx.getTailPos?.getD 0)
      InfoTreeNode.leanInfo DeclType.tactic none none text startPos endPos none filteredChildren
    | _ => .other childNodes
  | .hole _ => .hole

partial def removeOtherAndHoles (node : InfoTreeNode) : Option InfoTreeNode :=
  match node with
  | .context child =>
    match removeOtherAndHoles child with
    | some newChild => some (.context newChild)
    | none => none
  | InfoTreeNode.leanInfo decType name docString text startPos endPos namespc children =>
    let newChildren := children.map removeOtherAndHoles |>.filterMap id
    some (InfoTreeNode.leanInfo decType name docString text startPos endPos namespc newChildren)
  | .other children =>
    let newChildren := children.map removeOtherAndHoles |>.filterMap id
    if newChildren.isEmpty then
      none
    else
      some (.other newChildren)
  | .hole => none

partial def filterChildrenAtLevel (node : InfoTreeNode) (level : Nat) : Option InfoTreeNode :=
  match node with
  | .context child =>
    match filterChildrenAtLevel child level with
    | some newChild => some (.context newChild)
    | none => none
  | InfoTreeNode.leanInfo decType name docString text startPos endPos namespc children =>
    if level == 0 then
      some (InfoTreeNode.leanInfo decType name docString text startPos endPos namespc #[])
    else
      let newChildren := children.map fun child =>
        filterChildrenAtLevel child (level - 1)
      let filteredChildren := newChildren.filterMap id
      some (InfoTreeNode.leanInfo decType name docString text startPos endPos namespc filteredChildren)
  | .other children =>
    let newChildren := children.map fun child =>
      filterChildrenAtLevel child level
    let filteredChildren := newChildren.filterMap id
    if filteredChildren.isEmpty then
      none
    else
      some (.other filteredChildren)
  | .hole => none

/-- Helper: parse tactics in the current context -/
unsafe def parseInCurrentContext (input : String) (filePath : Option String := none) (chkptState : Option Command.State := none) : IO CheckpointedParseResult := do
  try
    --let inputCtx := Parser.mkInputContext input "<input>"
    let (initialCmdState, cmdState, messages, trees) ← try
      IO.processInput input chkptState Options.empty filePath
    catch e =>
      let errorInfo := ErrorInfo.mk (s!"Error during processing input: {e}") { line := 0, column := 0 }
      let parseResult : ParseResult := { trees := #[], errors := #[errorInfo] }
      return { parseResult := parseResult, chkptState := chkptState }


    -- Print any messages
    -- IO.println "\n=== Elaboration Messages ==="
    let mut errorInfos : Array ErrorInfo := #[]
    for msg in messages do
      if msg.severity == .error then
        let msgPos := Position.mk msg.pos.line msg.pos.column
        let errorInfo := ErrorInfo.mk (← msg.data.toString) msgPos
        errorInfos := errorInfos.push errorInfo
        -- IO.println s!"[ERROR] {← msg.data.toString} {msg.pos}"

    --   IO.println s!"[{severity}] {← msg.data.toString}"
    -- IO.println "=== End Messages ===\n"

    -- Print the cmdState environment
    -- IO.println "\n=== cmdState Environment Messages ==="
    -- for msg in cmdState.messages.toArray do
    --   let severity := match msg.severity with
    --     | .error => "ERROR"
    --     | .warning => "WARNING"
    --     | .information => "INFO"
    --   IO.println s!"[{severity}] {← msg.data.toString}"
    -- IO.println "=== End cmdState Messages ===\n"

    let level := 4 -- Only keep direct children of tactics
    let transformed_trees := trees.map (fun t =>
      let ans := removeOtherAndHoles (infoTreeToNode input t)
      let ans_d := ans.getD (.other #[])
      filterChildrenAtLevel ans_d level)
    let parseResult : ParseResult := { trees := transformed_trees, errors := errorInfos }
    return { parseResult := parseResult, chkptState := cmdState }
  catch e =>
    let errorInfo := ErrorInfo.mk (s!"Error in parseInCurrentContext: {e}") { line := 0, column := 0 }
    let parseResult : ParseResult := { trees := #[], errors := #[errorInfo] }
    return { parseResult := parseResult, chkptState := chkptState }

/-- Parse Lean code WITH elaboration to get InfoTrees (lightweight, no compilation!)

    Initializes Lean from current working directory (finds .lake/build automatically).
    For project-specific parsing, start the process from the project directory.
-/
unsafe def parseTacticsWithElaboration (input : String) (filePath : Option String := none) (chkptState : Option Command.State := none) : IO CheckpointedParseResult := do
  try
    -- Initialize Lean from current directory (finds .lake/build if present)
    Lean.initSearchPath (← Lean.findSysroot)
    Lean.enableInitializersExecution
    return ← parseInCurrentContext input filePath chkptState
  catch e =>
    let errorInfo := ErrorInfo.mk (s!"Error in parseTacticsWithElaboration: {e}") { line := 0, column := 0 }
    let parseResult : ParseResult := { trees := #[], errors := #[errorInfo] }
    return { parseResult := parseResult, chkptState := chkptState }

/-- Parse Lean code and extract all tactics (uses elaboration-based approach) -/
@[implemented_by parseTacticsWithElaboration]
opaque parseTactics (input : String) (filePath : Option String := none) (chkptState : Option Command.State := none) : IO CheckpointedParseResult

-- -- Test case 1: Simple proof with apply and exact
def simple_example := "theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
have h1 : p ∧ q := by
    sorry
apply And.intro
exact hq
exact hp
"

def more_complex_example := "theorem test3 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
have htemp : p ∧ q := by
    apply And.intro
    exact hp
    exact hq
simp [htemp]
rw [hp]
"

def import_example := "import Lean

theorem test_import (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq

"

def wrong_tactic_example := "theorem test_wrong (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
applly And.intro
exact hp
apply And.intro
"

def wrong_tactic_example2 := "theorem wrong_decl : Nat := by assdfadfs"



#eval parseTactics more_complex_example

#eval parseTactics import_example

#eval parseTactics wrong_tactic_example

#eval parseTactics wrong_tactic_example2

end TacticParser
