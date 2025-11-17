/-
Proof extractor for theorem, lemma, and example declarations.
Uses a validation-based approach: tests candidate delimiters by replacing
the proof with `sorry` and checking if it parses successfully.
-/
import Lean
import Lean.Parser
import TacticParser.Types
import TacticParser.SyntaxWalker
import Lean.Elab.Frontend

namespace TacticParser

open Lean Parser Elab

/-- Represents a candidate delimiter position and type -/
structure DelimiterCandidate where
  position : Nat  -- Byte position in the text
  delimiterType : String  -- ":=", "where", or "|"
  deriving Repr, BEq

instance : Ord DelimiterCandidate where
  compare a b := compare a.position b.position

/-- Try to parse a text snippet and return true if it parses without errors -/
def tryParseSuccessfully (text : String) (cmdState : Command.State): IO Bool := do
  let chkpt_result ← parseTactics text none (some cmdState)
  let parse_res := chkpt_result.1
  let new_cmd_state := chkpt_result.2
  pure (parse_res.errors.size == 0 ∧ new_cmd_state.isSome)

/-- Check if a substring starts with a given substring at a specific position -/
def substringStartsWith (text : String) (startPos : Nat) (substr : String) : Bool :=
  if startPos + substr.length > text.length then
    false
  else
    let extracted := text.drop startPos
    extracted.startsWith substr

/-- Find all occurrences of a substring in a text -/
def findSubstrOccurences (text : String) (substr : String) : List Nat :=
  if substr.length > text.length then
    []
  else
    let all_pos := List.range (text.length - substr.length + 1)
    let all_occurences := all_pos.filter (fun pos => substringStartsWith text pos substr)
    all_occurences

/-- Find all candidate delimiter positions -/
def findCandidateDelimiters (text : String) : List DelimiterCandidate :=
  let assignPositions := findSubstrOccurences text ":="
  let wherePositions := findSubstrOccurences text "where"
  let pipePositions := findSubstrOccurences text "|"

  let assignCandidates := assignPositions.map fun pos =>
    { position := pos, delimiterType := ":=" }
  let whereCandidates := wherePositions.map fun pos =>
    { position := pos, delimiterType := "where" }
  let pipeCandidates := pipePositions.map fun pos =>
    { position := pos, delimiterType := "|" }
  let assignCandidatesSorted := assignCandidates.toArray.qsort (fun a b => a.position < b.position)
  let whereCandidatesSorted := whereCandidates.toArray.qsort (fun a b => a.position < b.position)
  let pipeCandidatesSorted := pipeCandidates.toArray.qsort (fun a b => a.position < b.position)

  let allCandidates := assignCandidatesSorted.toList ++ whereCandidatesSorted.toList ++ pipeCandidatesSorted.toList
  allCandidates

def is_proof_extraction_needed (declInfo : DeclInfo) : Bool :=
  declInfo.declType == DeclType.theorem ||
  declInfo.declType == DeclType.lemma ||
  declInfo.declType == DeclType.example

def get_content_before_decl (fileContent : String) (declLineNum : Nat) : String :=
  let lines := fileContent.splitOn "\n"
  let beforeLines := lines.take (declLineNum - 1)
  (String.intercalate "\n" beforeLines) ++ "\n"

def get_in_between_content (fileContent : String) (startLine : Nat) (endLine : Nat) : String :=
  let lines := fileContent.splitOn "\n"
  let betweenLines := (lines.take endLine).drop (startLine - 1)
  String.intercalate "\n" betweenLines

/-- Extract proof from a declaration by testing candidate delimiters -/
unsafe def extractProofFromDecl
  (declInfo : DeclInfo)
  (cmdState : Command.State) : IO DeclInfo := do
  -- Only process theorem, lemma, example
  if !is_proof_extraction_needed declInfo then
    panic! s!"extractProofFromDecl called on non-proof decl: {declInfo.declType}"

  let text := declInfo.text

  -- Convert Position to byte offset
  -- Find all candidate delimiters
  let candidates := findCandidateDelimiters text

  -- Test each candidate
  for candidate in candidates do
    let beforeDelimiter := text.take candidate.position

    -- IO.println s!"beforeDelimiter:\n{beforeDelimiter}\n---"

    -- Build test text with full context
    let statementOnly := match candidate.delimiterType with
      | "|" => beforeDelimiter.trim ++ " := sorry"
      | ":=" => beforeDelimiter ++ " := sorry"
      | "where" => beforeDelimiter ++ " := sorry"
      | _ => beforeDelimiter ++ " := sorry"

    -- IO.println s!"statementOnly:\n{statementOnly}\n---"

    -- Try to parse
    let success ← tryParseSuccessfully statementOnly cmdState
    if success then
      -- Found valid split!
      -- IO.println s!"Found valid split at position {candidate.position} with delimiter {candidate.delimiterType}"
      let proof := text.drop candidate.position
      let thrm := text.take candidate.position
      return { declInfo with proof := some proof.trim , text := thrm.trim }
    -- else
    --  IO.println s!"Failed split at position {candidate.position} with delimiter {candidate.delimiterType}"

  -- No valid split found - no proof
  return { declInfo with proof := none }

unsafe def parse_between
(fileContent : String)
(decl: DeclInfo)
(prev: Nat)
(next: Nat)
(cmd_state : Option Command.State)
: IO (Nat × CheckpointedParseResult) := do
    -- IO.println s!"Extracting proof for segment between lines {prev} and {next}"
    let contextBeforeDecl := get_in_between_content fileContent prev next
    -- IO.println s!"Processing declaration at line {decl.startPos.line}-{decl.endPos.line}"
    -- IO.println s!"Declaration text:\n{decl.text}\n---"
    -- IO.println s!"--- Context Before Decl ----\n{contextBeforeDecl}\n--- Context Before Decl ----\n"
    let chkpt_parse_res ← parseTactics contextBeforeDecl none cmd_state
    return (next, chkpt_parse_res)

/-- Extract proofs from multiple declarations -/
unsafe def extractProofsFromDecls (decls : Array DeclInfo) (fileContent : String) : IO (Array DeclInfo) := do
  let mut result := #[]
  let mut prev := 0
  let mut cmd_state : Option Command.State := none
  let mut next := 0
  for decl in decls do
    if is_proof_extraction_needed decl then
      next := decl.startPos.line - 1
      let tup ← parse_between fileContent decl prev next cmd_state
      let chkpt_parse_res := tup.2
      -- IO.println s!"--- Context Before Decl ----\n{contextBeforeDecl}\n--- Context Before Decl ----\n"
      let parse_res := chkpt_parse_res.1
      let mut decl_fixed := decl
      if parse_res.errors.size > 0 then
        -- we should reparse with 1 less line to avoid partial lines
        let last_line_num := decl.startPos.line - 1
        let new_decl := { decl with startPos := { decl.startPos with line := last_line_num } }
        next := last_line_num - 1
        -- IO.println s!"Reparsing context before declaration at line {decl.startPos.line} with one less line"
        let tup ← parse_between fileContent decl prev next cmd_state
        let chkpt_parse_res := tup.2
        let mut missing_decl_text := get_in_between_content fileContent last_line_num last_line_num
        missing_decl_text := missing_decl_text.trim
        decl_fixed := { new_decl with text := missing_decl_text ++ "\n" ++ decl.text }
        if chkpt_parse_res.parseResult.errors.size > 0 then
          IO.println s!"Reparsing failed again for context before declaration at line {decl.startPos.line}-{decl.endPos.line}"
          IO.println s!"Errors:"
          for error in chkpt_parse_res.parseResult.errors do
            IO.println s!"Error while parsing context before declaration at line {decl.startPos.line}:
              {error.message} at {error.position.line}:{error.position.column}"
          IO.println s!"--- Context Before Decl ----\n{get_in_between_content fileContent prev next}\n--- Context Before Decl ----\n"
          IO.println s!"Decl text:\n{decl_fixed.text}\n---"
          panic! "Failed to parse context before declaration after reparsing"
        cmd_state := chkpt_parse_res.2
        prev := next
        next := tup.1
      else
        cmd_state := chkpt_parse_res.2
        prev := next
        next := tup.1
      -- for error in parse_res.errors do
      --   IO.println s!"Error while parsing context before declaration at line {decl.startPos.line}: {error.message} at {error.position.line}:{error.position.column}"
      -- if parse_res.errors.size > 0 then
      --   IO.println s!"--- Context Before Decl ----\n{contextBeforeDecl}\n--- Context Before Decl ----\n"
      --   IO.println s!"Failed to parse context before declaration at line {decl.startPos.line}"
      --   panic! "Failed to parse context"
      -- IO.println s!"Context before decl:\n{contextBeforeDecl}\n---"
      -- let cmd_state_error_pair ← get_cmd_state contextBeforeDecl cmd_state
      -- cmd_state := cmd_state_error_pair.1
      -- let errors := cmd_state_error_pair.2
      -- for err in errors do
      --   IO.println s!"Error while updating cmd_state: {err.message} at {err.position.line}:{err.position.column}"
      -- if cmd_state.isNone ∨ errors.size > 0 then
      --   IO.println s!"Failed to update cmd_state before processing declaration at line {decl.startPos.line}"
      --   panic! "Failed to update cmd_state"
      -- else
      let cmd_st ← match cmd_state with
        | some st => pure st
        | none => panic! "Failed to get valid cmd_state before processing declaration"
      let processed ← extractProofFromDecl decl_fixed cmd_st
      result := result.push processed
    else
      result := result.push decl
  return result

end TacticParser
