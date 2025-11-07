import Lean
import Lean.Data.Json
import TacticParser.Types
namespace TacticParser

open Lean
open Lean.Parser

/-- Identify the type of declaration from syntax -/
partial def identifySomeDeclType (stx : Syntax) : Option (DeclType × Nat) :=
  let kind := stx.getKind
  -- Check if this is a declaration wrapper, if so, look inside
  if kind == `Lean.Parser.Command.declaration then
    match stx with
    | Syntax.node _ _ args =>
      -- Look for the actual declaration type in the children
      -- NOTE: using findIdx? for backward compatibility to Lean 4.15
      -- (as.findIdx? p).getD as.size
      let idx := (args.findIdx? (fun a => (identifySomeDeclType a).isSome)).getD args.size;
      if idx = args.size then
        some (.unknown, idx)
      else
        let decl := args[idx]!
        let declTypeOpt := identifySomeDeclType decl
        match declTypeOpt with
        | some dt => some (dt.1, idx)
        | none => some (.unknown, idx)
    | _ => none
  else if kind == `Lean.Parser.Command.end then some (.end, 0)
  else if kind == `Lean.Parser.Command.namespace then some (.namespace, 0)
  else if kind == `Lean.Parser.Command.inductive then some (.inductive, 0)
  else if kind == `Lean.Parser.Command.theorem then some (.theorem, 0)
  else if kind == `Lean.Parser.Command.definition then some (.def, 0)
  else if kind == `Lean.Parser.Command.axiom then some (.axiom, 0)
  else if kind == `Lean.Parser.Command.structure then some (.structure, 0)
  else if kind == `Lean.Parser.Command.classDecl then some (.class_decl, 0)
  else if kind == `Lean.Parser.Command.instance then some (.instance, 0)
  else if kind == `Lean.Parser.Command.example then some (.example, 0)
  else if kind == `Lean.Parser.Command.otherDecl then some (.other, 0)
  else none

/-- Identify the type of declaration from syntax -/
unsafe def identifyDeclType (stx : Syntax) : DeclType :=
  match identifySomeDeclType stx with
  | some dt => dt.1
  | none => .unknown

/-- Check if a syntax node is an attribute/modifier that should be skipped -/
def isModifierOrAttribute (stx : Syntax) : Bool :=
  let kind := stx.getKind
  -- Skip attributes (@[...]), docstrings, and other modifiers
  kind == `Lean.Parser.Term.attrInstance ||
  kind == `Lean.Parser.Command.docComment ||
  kind == `Lean.Parser.Term.attributes ||
  kind.toString.startsWith "Lean.Parser.Command.declModifiers"

/-- Extract the name of the declaration from syntax tree -/
partial def extractDeclName (stx : Syntax) : String :=
  match stx with
  | Syntax.ident _ _ name _ => name.toString
  | Syntax.node _ kind args =>
    -- For declaration nodes, skip modifiers/attributes and keywords to find the name
    if kind == `Lean.Parser.Command.declaration ||
       kind == `Lean.Parser.Command.theorem ||
       kind == `Lean.Parser.Command.definition ||
       kind == `Lean.Parser.Command.inductive ||
       kind == `Lean.Parser.Command.structure ||
       kind == `Lean.Parser.Command.classDecl ||
       kind == `Lean.Parser.Command.instance ||
       kind == `Lean.Parser.Command.axiom then
      -- Skip attributes and find the first identifier that's not a keyword
      (args.findSome? fun arg =>
        if isModifierOrAttribute arg then
          none
        else
          let result := extractDeclName arg
          -- Skip keywords like "theorem", "def", etc.
          if result != Name.anonymous.toString &&
             result != "theorem" &&
             result != "def" &&
             result != "lemma" &&
             result != "inductive" &&
             result != "structure" &&
             result != "class" &&
             result != "instance" &&
             result != "axiom" then
            some result
          else
            none
      ).getD Name.anonymous.toString
    else
      -- For other nodes, search through arguments
      (args.findSome? fun arg =>
        let result := extractDeclName arg
        if result != Name.anonymous.toString then some result else none
      ).getD Name.anonymous.toString
  | _ => Name.anonymous.toString

/-- Comment parsing state machine, can parse nested comments too -/
partial def trimComment (text : String) (state : Nat := 0) (depth : Nat := 0) : Nat :=
  if text.startsWith "--" ∧ state == 0 then
    -- we are not inside a block comment, so this is a line comment
    let newState := 0
    -- Go till the end of line
    let endOfLine := text.find (fun c => c == '\n')
    let remaining := text.drop endOfLine.byteIdx
    let ep := trimComment remaining newState depth
    endOfLine.byteIdx + ep
  else if text.startsWith "/-" ∧ state == 0 then
    -- starting of a block comment
    let newState := 1
    let remaining := text.drop 2
    let ep := trimComment remaining newState (depth + 1)
    ep + 2
  else if text.startsWith "-/" ∧ state == 1 then
    -- ending of a block comment
    let newDepth := depth - 1
    let newState := if newDepth == 0 then 0 else 1
    let remaining := text.drop 2
    let ep := trimComment remaining newState newDepth
    ep + 2
  else if text.length == 0 then
    0
  else
    -- consume one character and continue
    if state == 0 ∧ text.trimLeft.length == text.length then
      -- not in comment and no leading spaces, stop
      0
    else
      let remaining := text.drop 1
      let ep := trimComment remaining state depth
      ep + 1

def comment_testcase := "
/- This is a /* nested */ comment -/
/--This is an example lemma-/ --- let's see how it works
def exampleLemma : Nat := 42
"

#eval trimComment comment_testcase  -- should return length of comment part
#eval comment_testcase.drop (trimComment comment_testcase)  -- should return " rest of code"

def no_comment_testcase := "def noComment : Nat := 100"

#eval trimComment no_comment_testcase  -- should return 0
#eval no_comment_testcase.drop (trimComment no_comment_testcase)  -- should return "def noComment : Nat := 100"

def postProcess (text : String) : String × List Nat :=
  -- Replace lines with `^lemma ` with `theorem `
  let lines := text.splitOn "\n"
  let processedLines := lines.mapIdx fun i line =>
    if line.trimLeft.startsWith "lemma " then
      let leadingSpaces := line.takeWhile (fun c => c == ' ' ∨ c == '\t')
      let newLine := leadingSpaces ++ "theorem " ++ line.trimLeft.drop "lemma ".length
      (newLine, lines.length)
      --(newLine, lines.length)
    else
      (line, i)
  let linesOnly := processedLines.map Prod.fst
  let lineNumbers := processedLines.map Prod.snd
  let filteredLineNums := lineNumbers.filter (fun n => n != lines.length)
  (String.intercalate "\n" linesOnly, filteredLineNums)

unsafe def parseCommon
  (originalContent : String)
  (parserState : ModuleParserState)
  (pmctx : ParserModuleContext)
  (inputCtx : InputContext)
  : IO (Array DeclInfo) := do
  -- First pass: parse all commands and collect their positions
  -- We parse the ORIGINAL content to find declaration boundaries
  let mut commands : Array (String.Pos × Syntax) := #[]
  let mut pstate := parserState
  let mut done := false

  while !done do
    let startPos := pstate.pos
    -- IO.println s!"Parsing at position: {pstate.pos}, kind will be: ..."

    -- Try to parse a command from original content
    let (stx, pstate', msgs) := parseCommand inputCtx pmctx pstate {}
    pstate := pstate'

    -- IO.println s!"  Got kind: {stx.getKind}"
    -- IO.println s!"  New position: {pstate.pos}, atEnd: {inputCtx.atEnd pstate.pos}, messages: {msgs.toList.length}"

    -- Store command with its start position
    commands := commands.push (startPos, stx)

    -- Check if we made progress or reached end
    if pstate.pos == startPos then --|| inputCtx.atEnd pstate.pos then
      -- IO.println s!"  Stopping: pos unchanged={pstate.pos == startPos}, atEnd={inputCtx.atEnd pstate.pos}"
      done := true

  -- Second pass: extract and re-parse declarations
  let mut decls : Array DeclInfo := #[]
  let mut openNamespaces : List String := []
  for i in [:commands.size] do
    let (parsePos, stx) := commands[i]!

    -- Get position range for this command
    let realStart := match stx.getRange? with
      | some range => range.start
      | none => parsePos
    let endPos := if i + 1 < commands.size then
      let (nextParsePos, nextStx) := commands[i + 1]!
      let nextRealStart := match nextStx.getRange? with
        | some range => range.start
        | none => nextParsePos
      ⟨nextRealStart.byteIdx - 1⟩
    else
      ⟨originalContent.endPos.byteIdx⟩

    -- Extract text from ORIGINAL content
    let text := originalContent.extract realStart endPos

    -- Strip comments to check if this starts with "lemma"
    let commentEnd := trimComment text
    let docStringStr := (text.take commentEnd).trim
    let mut docString := none
    if !docStringStr.isEmpty then
      docString := some docStringStr
    let textWithoutComments := text.drop commentEnd
    let isLemma := textWithoutComments.startsWith "lemma "

    -- Print the docstring and the text without comments for debugging
    -- IO.println s!"Docstring part:\n{docString}\n--- End of docstring ---"
    -- IO.println s!"Text without comments:\n{textWithoutComments}\n--- End of text without comments ---"
    -- -- print if it's identified as lemma
    -- IO.println s!"Is lemma: {isLemma}"

    let mut textToParse := text
    if isLemma then
    -- If it's a lemma, preprocess it for parsing
      -- replace the "lemma" at the end position of the comment with "theorem"
      textToParse := text.take commentEnd ++
                     "theorem " ++
                     textWithoutComments.drop "lemma ".length

    let declInputCtx := mkInputContext textToParse "<input>"
    let (_, declParserState, _) ← parseHeader declInputCtx
    let (declStx, _, _) := parseCommand declInputCtx pmctx declParserState {}

    -- Now identify the declaration type from the (possibly preprocessed) syntax
    let declType := identifyDeclType declStx
    let name := extractDeclName declStx

    if declType == .namespace then
      openNamespaces := openNamespaces.append [name]
    else if declType == .end then
      -- Pop the last opened namespace if any
      openNamespaces := openNamespaces.dropLast

    -- If we preprocessed it and it parsed as theorem, it's actually a lemma
    let actualDeclType := if isLemma && declType == .theorem then .lemma else declType
    let namespc :=
      if openNamespaces.isEmpty then
        none
      else
        some (String.intercalate "." openNamespaces)

    let start_pos := get_position_from_char_pos originalContent realStart.byteIdx
    let end_pos := get_position_from_char_pos originalContent endPos.byteIdx

    let info : DeclInfo := {
      declType := actualDeclType
      name := name
      startPos := start_pos
      endPos := end_pos
      text := textWithoutComments -- Store text after extracting docstring
      docString := docString -- Store extracted docstring
      namespc := namespc
    }
    decls := decls.push info

  return decls

/-- Parse a Lean 4 file and extract declaration information -/
unsafe def parseDecls (originalContent : String) : IO (Array DeclInfo) := do
  let (postProcessedContent, modifiedLineIdx) := postProcess originalContent
  let inputCtx := mkInputContext postProcessedContent "<input>"

  -- Parse the header (using original content)
  let (_, parserState, _) ← parseHeader inputCtx

  -- Create a minimal parser context with empty environment
  let env ← Lean.importModules #[] {} 0
  let opts := {}
  let pmctx : ParserModuleContext := {
    env := env
    options := opts
  }
  let decls ← parseCommon postProcessedContent parserState pmctx inputCtx

  return decls

/-- Parse a Lean 4 file and extract declaration information -/
unsafe def parseFile (filepath : System.FilePath) : IO (Array DeclInfo) := do
  let originalContent ← IO.FS.readFile filepath
  let (postProcessedContent, modifiedLineIdx) := postProcess originalContent
  let inputCtx := mkInputContext postProcessedContent filepath.toString

  -- Parse the header (using original content)
  let (_, parserState, _) ← parseHeader inputCtx

  -- Create a minimal parser context with empty environment
  let env ← Lean.importModules #[] {} 0
  let opts := {}
  let pmctx : ParserModuleContext := {
    env := env
    options := opts
  }

  let decls ← parseCommon postProcessedContent parserState pmctx inputCtx

  return decls

/-- Convert DeclInfo to JSON -/
def declInfoToJson (info : DeclInfo) : Json :=
  let baseFields := [
    ("declType", Json.str (toString info.declType)),
    ("name", Json.str info.name),
    ("startPos", toJson info.startPos),
    ("endPos", toJson info.endPos),
    ("text", Json.str info.text)
  ]
  let withDocString := match info.docString with
    | some doc => baseFields ++ [("docString", Json.str doc)]
    | none => baseFields
  Json.mkObj withDocString

/-- Simple helper to print declaration info -/
def printDeclInfo (info : DeclInfo) : IO Unit := do
  IO.println s!"[{info.declType}] {info.name}"
  IO.println s!"  Position: {toJson info.startPos} - {toJson info.endPos}"
  let preview := if info.text.length > 100 then
    info.text.take 50 ++ "\n ... more text ... \n" ++ info.text.drop (info.text.length - 50)
  else
    info.text
  IO.println s!"  Text: {preview}"
  IO.println ""

/-- Export declarations to JSON file -/
def exportToJson (decls : Array DeclInfo) (outputPath : System.FilePath) : IO Unit := do
  let jsonArray := Json.arr (decls.map declInfoToJson)
  let jsonStr := jsonArray.pretty
  IO.FS.writeFile outputPath jsonStr
  IO.println s!"Exported {decls.size} declaration(s) to {outputPath}"

/-- Parse and print all declarations in a file -/
unsafe def parseAndPrint (filepath : System.FilePath) : IO Unit := do
  IO.println s!"Parsing file: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))

  let decls ← parseFile filepath

  if decls.isEmpty then
    IO.println "No declarations found."
  else
    IO.println s!"Found {decls.size} declaration(s):"
    IO.println ""
    for decl in decls do
      printDeclInfo decl

/-- Parse file and export to both console and JSON -/
unsafe def parseAndExport (filepath : System.FilePath) (jsonOutput : Option System.FilePath := none) : IO Unit := do
  let decls ← parseFile filepath

  -- Print to console
  IO.println s!"Parsing file: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))

  if decls.isEmpty then
    IO.println "No declarations found."
  else
    IO.println s!"Found {decls.size} declaration(s):"
    IO.println ""
    for decl in decls do
      printDeclInfo decl

  -- Export to JSON if output path provided
  match jsonOutput with
  | some outPath => exportToJson decls outPath
  | none => pure ()

def test_str := "import Mathlib
namespace Lean4Proj1

def hello := \"world\"
theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp


theorem test2 : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩


-- show a proof which uses calc
theorem test_calc (n: Nat) : n^2 + 2*n + 1 = (n + 1)*(n + 1) := by
calc
    _ = n^2 + n*2 + 1 := by rw [Nat.mul_comm 2 n]
    _ = n^2 + (n + n) + 1 := by rw [Nat.mul_two]
    _ = n^2 + n + n + 1 := by rw [←Nat.add_assoc]
    _ = n*n + n + n + 1 := by rw [Nat.pow_two]
    _ = n*n + n*1 + n + 1 := by rw [Nat.mul_one n]
    _ = n*(n + 1) + n + 1 := by rw [Nat.left_distrib n n 1]
    _ = n*(n + 1) + (n + 1) := by rw [Nat.add_assoc]
    _ = n*(n + 1) + 1*(n + 1) := by rw (config := { occs := .pos [2]}) [←Nat.mul_one (n + 1), Nat.mul_comm]
    _ = (n + 1)*(n + 1) := by rw [Nat.right_distrib n 1 (n + 1)]
done

end Lean4Proj1

namespace Lean4Proj2

example : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩

/-- This is a test theorem -/
theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp
done

@[simp]
theorem test3 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp

theorem imo_1959_p1
  (n : ℕ)
  (h₀ : 0 < n) :
  Nat.gcd (21*n + 4) (14*n + 3) = 1 := by
rw [Nat.gcd_rec]
rw [Nat.mod_eq_of_lt (by linarith)]
rw [Nat.gcd_rec]
rw [Nat.gcd_rec]
have eq₂ : (21 * n + 4) % (14 * n + 3) = 7 * n + 1 := by
    have eq₁ : 21 * n + 4 = (14 * n + 3) + (7 * n + 1) := by ring
    rw [eq₁, Nat.add_mod, Nat.mod_self, zero_add]
    have h₂ : 7 * n + 1 < 14 * n + 3 := by linarith
    rw [Nat.mod_eq_of_lt]
    rw [Nat.mod_eq_of_lt]
    exact h₂
    rw [Nat.mod_eq_of_lt]
    exact h₂
    exact h₂
rw [eq₂]
sorry


lemma pow_dvd_pow (a : α) (h : m ≤ n) : a ^ m ∣ a ^ n :=
  ⟨a ^ (n - m), by rw [← pow_add, Nat.add_comm, Nat.sub_add_cancel h]⟩

lemma dvd_pow (hab : a ∣ b) : ∀ {n : ℕ} (_ : n ≠ 0), a ∣ b ^ n
  | 0,     hn => (hn rfl).elim
  | n + 1, _  => by rw [pow_succ']; exact hab.mul_right _

alias Dvd.dvd.pow := dvd_pow

lemma dvd_pow_self (a : α) {n : ℕ} (hn : n ≠ 0) : a ∣ a ^ n := dvd_rfl.pow hn

end Lean4Proj2
"

#eval parseDecls test_str

#eval (test_str.extract ⟨15⟩ ⟨37⟩)

#eval (test_str.extract ⟨37⟩ ⟨58⟩)

#eval (test_str.extract ⟨298⟩ ⟨912⟩)

#eval get_position_from_char_pos test_str 57 -- expect line 4, column 20

#eval get_position_from_char_pos test_str 299 -- line 18, column 1

end TacticParser
