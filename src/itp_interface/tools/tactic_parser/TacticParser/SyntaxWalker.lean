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

def nodeIsHole (node : InfoTreeNode) : Bool :=
  match node with
  | .hole => true
  | _ => false

def nodeEndPos (node : InfoTreeNode) : Option Position :=
  match node with
  | InfoTreeNode.leanInfo _ _ _ _ _ endPos _ _ => some endPos
  | _ => none

partial def filterAllNodesWhichDontStartAndEndOnLine (node : InfoTreeNode) (line_num: Nat) : Array InfoTreeNode :=
match node with
| .context child =>
  filterAllNodesWhichDontStartAndEndOnLine child line_num
| InfoTreeNode.leanInfo decType name docString text startPos endPos namespc children =>
  let newChildren := children.flatMap fun child =>
    filterAllNodesWhichDontStartAndEndOnLine child line_num
  if startPos.line != line_num ∨ endPos.line != line_num then
    newChildren
  else
    -- Add self to the front of the list
    #[InfoTreeNode.leanInfo decType name docString text startPos endPos namespc #[]] ++ newChildren
| .other children =>
  children.flatMap fun child =>
    filterAllNodesWhichDontStartAndEndOnLine child line_num
| .hole => #[]

def getMaxLineExtent (node : InfoTreeNode) (line_num: Nat) : InfoTreeNode × Nat :=
let all_possible_nodes := filterAllNodesWhichDontStartAndEndOnLine node line_num
let arg_max := all_possible_nodes.foldl (fun (acc_node, acc_len) n =>
  let len := (match n with
    | InfoTreeNode.leanInfo _ _ _ _ startPos endPos _ _ =>
      endPos.column - startPos.column
    | _ => 0
  )
  if len > acc_len then
    (n, len)
  else
    (acc_node, acc_len)
) (InfoTreeNode.hole, 0)
arg_max

partial def getAllLinesInTree (node : InfoTreeNode) : Std.HashSet Nat :=
  match node with
  | .context child =>
    getAllLinesInTree child
  | InfoTreeNode.leanInfo _ _ _ _ startPos endPos _ children =>
    let childrenLines := children.map getAllLinesInTree
    (childrenLines.foldl (init := ({}: Std.HashSet Nat)) (fun acc lines =>
      acc.union lines)).union {startPos.line, endPos.line}
  | .other children =>
    let childrenLines := children.map getAllLinesInTree
    childrenLines.foldl (init := {}) (fun acc lines =>
      acc.union lines)
  | .hole => {}

def getAllLineNumsFromTrees (trees : Array InfoTreeNode) : Array Nat :=
(trees.foldl (init := ({}: Std.HashSet Nat)) (fun acc tree =>
acc.union (getAllLinesInTree tree))).toArray.insertionSort

def getAllExtents (trees : Array InfoTreeNode) : Array InfoTreeNode :=
let line_nums := getAllLineNumsFromTrees trees
(line_nums.flatMap (
  fun line_num =>
    (trees.foldl (fun acc tree =>
      let (n, _) := getMaxLineExtent tree line_num
      acc.push n
    ) (#[] : Array InfoTreeNode)).insertionSort (fun n1 n2 =>
      match (nodeEndPos n1, nodeEndPos n2) with
      | (some pos1, some pos2) => pos1.column < pos2.column
      | _ => false
    )
)).filter fun n => ¬ nodeIsHole n

def getTextFromPosition (input : String) (startPos : Position) (endPos : Position) : String :=
  let lines := input.splitOn "\n"
  if startPos.line > lines.length ∨ startPos.line == 0 ∨ endPos.line == 0 then
    ""
  else
    let relevantLines := (lines.take endPos.line).drop (startPos.line - 1)
    let firstLine := relevantLines[0]!.drop startPos.column--.extract ⟨startPos.column⟩ ⟨relevantLines[0]!.length⟩
    let lastLine := relevantLines[relevantLines.length - 1]!.take endPos.column
    let middleLines := (relevantLines.take (relevantLines.length - 1)).drop 1
    let actualLines := if relevantLines.length > 1 then [firstLine] ++ middleLines ++ [lastLine]  else [firstLine]
    String.intercalate "\n" actualLines

def dropNewLineAndCountSpaces (s : String) : String × String :=
  let strWithoutSpace := s.dropRightWhile (fun c => c == '\t' || c == ' ')
  let rightSpace := s.takeRightWhile (fun c => c == '\t' || c == ' ')
  (strWithoutSpace.trimRight, rightSpace)

/-- Helper: parse tactics in the current context -/
unsafe def parseInCurrentContext (input : String) (filePath : Option String := none) (chkptState : Option Command.State := none) : IO CheckpointedParseResult := do
  try
    --let inputCtx := Parser.mkInputContext input "<input>"
    let (initialCmdState, cmdState, messages, trees) ← try
      IO.processInput input chkptState Options.empty filePath
    catch e =>
      let errorInfo := ErrorInfo.mk (s!"Error during processing input: {e}") { line := 0, column := 0 }
      let parseResult : ParseResult := { trees := #[], errors := #[errorInfo] }
      return { parseResult := parseResult, chkptState := chkptState , lineNum := none }


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

    let level := 0 -- Only keep direct children of tactics
    -- let transformed_trees := trees.map (fun t =>
    --   let ans := removeOtherAndHoles (infoTreeToNode input t)
    --   let ans_d := ans.getD (.other #[])
    --   filterChildrenAtLevel ans_d level)
    let transformed_trees := trees.map (fun t => removeOtherAndHoles (infoTreeToNode input t))
    let t_trees := transformed_trees.map (fun t => t.getD (.other #[]))
    let lineExtents := getAllExtents t_trees
    let extentStruct := lineExtents.map getInfoNodeStruct
    -- Go over all line extents and reassign the end_pos of the next node
    let mut adjusted_trees : Array InfoNodeStruct := #[]
    for i in [1:lineExtents.size] do
      let prev_node := extentStruct[i - 1]!.getD default
      let curr_node := extentStruct[i]!.getD default
      let new_prev_node := {prev_node with endPos := curr_node.startPos}
      adjusted_trees := adjusted_trees.push new_prev_node

    let mut last_node := extentStruct[extentStruct.size - 1]!.getD default
    let lines := input.splitOn "\n"
    let lineCount := lines.length
    last_node := {last_node with endPos := { line := lineCount, column := lines.getLast!.length }}
    adjusted_trees := adjusted_trees.push last_node
    -- Fix the text fields based on updated positions
    adjusted_trees := adjusted_trees.map fun node =>
      let new_text := getTextFromPosition input node.startPos node.endPos
      { node with text := new_text }
    let mut (prev_text, right_space) := dropNewLineAndCountSpaces adjusted_trees[0]!.text
    adjusted_trees := adjusted_trees.set! 0 {adjusted_trees[0]! with text := prev_text}
    for i in [1:adjusted_trees.size] do
      let curr_node := adjusted_trees[i]!
      let mut (curr_text, curr_right_space) := dropNewLineAndCountSpaces curr_node.text
      curr_text := right_space ++ curr_text
      right_space := curr_right_space
      adjusted_trees := adjusted_trees.set! i {curr_node with text := curr_text}

    --   let new_prev_node := {prev_node with endPos := nodeEndPos curr_node.getD prev_node}
    let parseResult : ParseResult := { trees := adjusted_trees, errors := errorInfos }
    return { parseResult := parseResult, chkptState := cmdState , lineNum := lineCount }
  catch e =>
    let errorInfo := ErrorInfo.mk (s!"Error in parseInCurrentContext: {e}") { line := 0, column := 0 }
    let parseResult : ParseResult := { trees := #[], errors := #[errorInfo] }
    return { parseResult := parseResult, chkptState := chkptState , lineNum := none }

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
    return { parseResult := parseResult, chkptState := chkptState , lineNum := none }

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
have htemp : p ∧ q
:= by
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


def temp := (parseTactics more_complex_example)

#eval temp


#eval parseTactics import_example

#eval parseTactics wrong_tactic_example

#eval parseTactics wrong_tactic_example2

end TacticParser
