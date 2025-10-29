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

/-- Process commands with InfoTrees enabled (like REPL does) -/
def processCommandsWithInfoTrees
    (inputCtx : Parser.InputContext) (parserState : Parser.ModuleParserState)
    (commandState : Command.State) : IO (Command.State × Array Message × Array InfoTree) := do
  let commandState := { commandState with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState <&> Frontend.State.commandState
  pure (s, s.messages.toArray, s.infoState.trees.toArray)


/-- Extract the source text for a syntax node -/
def syntaxToString (stx : Syntax) : String :=
  stx.reprint.getD (toString stx)

-- /-- Check if string contains substring -/
-- def stringContains (s : String) (sub : String) : Bool :=
--   (s.splitOn sub).length > 1

-- /-- Check if a syntax kind is a tactic (not a sequence/grouping construct) -/
-- def isTacticKind (kind : Name) : Bool :=
--   let s := kind.toString
--   s.startsWith "Lean.Parser.Tactic." &&
--   !stringContains s "tacticSeq" &&
--   !stringContains s "Syntax" &&
--   kind != `null

-- /-- Extract individual tactics from syntax -/
-- partial def extractTactics (input : String) (stx : Syntax) : Array TacticInfo :=
--   match stx with
--   | .node _ kind children =>
--     -- Check if this is an individual tactic
--     if isTacticKind kind then
--       let text := syntaxToString stx |>.trim
--       if text.isEmpty || text == "by" || text == "done" then
--         -- Recurse into children for nested tactics
--         children.foldl (fun acc child => acc ++ extractTactics input child) #[]
--       else
--         let startPos := posToLineColumn input (stx.getPos?.getD 0)
--         let endPos := posToLineColumn input (stx.getTailPos?.getD 0)
--         let tacticInfo := { text, startPos, endPos : TacticInfo }
--         -- Also check children for nested tactics (e.g., in try/catch)
--         let childTactics := children.foldl (fun acc child => acc ++ extractTactics input child) #[]
--         if childTactics.isEmpty then
--           #[tacticInfo]
--         else
--           #[tacticInfo] ++ childTactics
--     else
--       -- Not a tactic node, recurse into children
--       children.foldl (fun acc child => acc ++ extractTactics input child) #[]
--   | _ => #[]

-- /-- Find all `by` blocks in the syntax tree -/
-- partial def findByBlocks (stx : Syntax) : Array Syntax :=
--   match stx with
--   | .node _ kind children =>
--     let result := if kind == `Lean.Parser.Term.byTactic then #[stx] else #[]
--     -- Recurse into children
--     children.foldl (fun acc child => acc ++ findByBlocks child) result
--   | _ => #[]

-- /-- Parse all syntax from input string, collecting partial parses -/
-- partial def parseAllSyntax (input : String) (env : Environment) : Array Syntax :=
--   let inputCtx := Parser.mkInputContext input "<input>"
--   let s : Parser.ModuleParserState := {}
--   let pmctx : Parser.ParserModuleContext := { env := env, options := {} }

--   let rec loop (state : Parser.ModuleParserState) (msgs : MessageLog) (acc : Array Syntax) : Array Syntax :=
--     if inputCtx.input.atEnd state.pos then
--       acc
--     else
--       let (cmd, state', msgs') := Parser.parseCommand inputCtx pmctx state msgs
--       -- Always add the command, even if incomplete
--       let acc' := acc.push cmd
--       -- Stop if position didn't advance (parser is stuck)
--       if state'.pos == state.pos then
--         acc'
--       else
--         loop state' msgs' acc'

--   loop s {} #[]



-- /-- Check if range1 fully contains range2 -/
-- def rangeContains (start1 end1 start2 end2 : Position) : Bool :=
--   -- Range1 contains range2 if:
--   -- start1 <= start2 AND end2 <= end1
--   let startBefore := start1.line < start2.line || (start1.line == start2.line && start1.column <= start2.column)
--   let endAfter := end1.line > end2.line || (end1.line == end2.line && end1.column >= end2.column)
--   startBefore && endAfter

-- /-- Get all leaf tactics (tactics that don't contain other tactics) -/
-- def getLeafTactics (tactics : Array TacticInfo) : Array TacticInfo :=
--   tactics.filter fun t =>
--     -- A tactic is a leaf if it doesn't contain any other tactic
--     !tactics.any fun child =>
--       -- Skip if it's the same tactic
--       if t.startPos == child.startPos && t.endPos == child.endPos then
--         false
--       else
--         -- Check if t contains child
--         rangeContains t.startPos t.endPos child.startPos child.endPos

-- /-- Remove exact duplicates (same text and position) -/
-- def deduplicateTactics (tactics : Array TacticInfo) : Array TacticInfo :=
--   let rec go (acc : Array TacticInfo) (remaining : Array TacticInfo) (idx : Nat) : Array TacticInfo :=
--     if idx >= remaining.size then
--       acc
--     else
--       let t := remaining[idx]!
--       -- Check if this tactic already exists in acc
--       let isDuplicate := acc.any fun existing =>
--         existing.text == t.text &&
--         existing.startPos == t.startPos &&
--         existing.endPos == t.endPos
--       if isDuplicate then
--         go acc remaining (idx + 1)
--       else
--         go (acc.push t) remaining (idx + 1)
--   go #[] tactics 0

-- /-- Check if text is just punctuation or other non-tactic token -/
-- def isNonTactic (text : String) : Bool :=
--   let text := text.trim
--   -- Filter out: empty, "by", "done", and single punctuation characters
--   text.isEmpty || text == "by" || text == "done" ||
--   (text.length == 1 && (
--     text == "]" || text == "[" || text == ")" || text == "(" ||
--     text == "}" || text == "{" || text == "," || text == ";" ||
--     text == ":" || text == "|" || text == "·"
--   ))

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
      .tacticInfo text startPos endPos filteredChildren
    | _ => .other childNodes
  | .hole _ => .hole

partial def removeOtherAndHoles (node : InfoTreeNode) : Option InfoTreeNode :=
  match node with
  | .context child =>
    match removeOtherAndHoles child with
    | some newChild => some (.context newChild)
    | none => none
  | .tacticInfo text startPos endPos children =>
    let newChildren := children.map removeOtherAndHoles |>.filterMap id
    some (.tacticInfo text startPos endPos newChildren)
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
  | .tacticInfo text startPos endPos children =>
    if level == 0 then
      some (.tacticInfo text startPos endPos #[])
    else
      let newChildren := children.map fun child =>
        filterChildrenAtLevel child (level - 1)
      let filteredChildren := newChildren.filterMap id
      some (.tacticInfo text startPos endPos filteredChildren)
  | .other children =>
    let newChildren := children.map fun child =>
      filterChildrenAtLevel child level
    let filteredChildren := newChildren.filterMap id
    if filteredChildren.isEmpty then
      none
    else
      some (.other filteredChildren)
  | .hole => none

/- Helper: parse tactics in the current context -/
unsafe def parseInCurrentContext (input : String) : IO ParseResult := do
  try
    let inputCtx := Parser.mkInputContext input "<input>"

    -- Parse header (imports)
    let (header, parserState, messages) ← Parser.parseHeader inputCtx
    let (env, _messages) ← processHeader header {} messages inputCtx 0
    let cmdState := Command.mkState env messages {}

    -- Process commands with elaboration (NO compilation!)
    let (_cmdState, _messages, trees) ← processCommandsWithInfoTrees inputCtx parserState cmdState

    -- Print any messages
    -- IO.println "\n=== Elaboration Messages ==="
    -- for msg in messages do
    --   let severity := match msg.severity with
    --     | .error => "ERROR"
    --     | .warning => "WARNING"
    --     | .information => "INFO"
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
    return { trees := transformed_trees, error := none }

    -- return { tree := treeNode, error := none }
  catch e =>
    return { trees := #[], error := some s!"Error in parseInCurrentContext: {e}" }

/-- Parse Lean code WITH elaboration to get InfoTrees (lightweight, no compilation!)

    Initializes Lean from current working directory (finds .lake/build automatically).
    For project-specific parsing, start the process from the project directory.
-/
unsafe def parseTacticsWithElaboration (input : String) : IO ParseResult := do
  try
    -- Initialize Lean from current directory (finds .lake/build if present)
    Lean.initSearchPath (← Lean.findSysroot)
    Lean.enableInitializersExecution
    return ← parseInCurrentContext input
  catch e =>
    return { trees := #[], error := some s!"Error in parseTacticsWithElaboration: {e}" }

/-- Parse Lean code and extract all tactics (uses elaboration-based approach) -/
@[implemented_by parseTacticsWithElaboration]
opaque parseTactics (input : String) : IO ParseResult

-- -- Test case 1: Simple proof with apply and exact
def simple_example := "theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
have h1 : p ∧ q := by
    sorry
apply And.intro
exact hq
exact hp"

def more_complex_example := "theorem test3 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
have htemp : p ∧ q := by
    apply And.intro
    exact hp
    exact hq
simp [htemp]
rw [hp]
"

#eval parseTacticsWithElaboration simple_example

#eval parseTacticsWithElaboration more_complex_example

end TacticParser
