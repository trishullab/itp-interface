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
  (cmdState : Command.State)
  (extra_content: Option String := none) : IO DeclInfo := do
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
    let mut statementOnly := match candidate.delimiterType with
      | "|" => beforeDelimiter.trim ++ " := sorry"
      | ":=" => beforeDelimiter ++ " := sorry"
      | "where" => beforeDelimiter ++ " := sorry"
      | _ => beforeDelimiter ++ " := sorry"


    -- If extra content is provided, prepend it
    statementOnly :=
      match extra_content with
      | some extra => extra ++ (if extra.endsWith "\n" then statementOnly else "\n" ++ statementOnly)
      | none => statementOnly

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
(prev: Nat)
(next: Nat)
(cmd_state : Option Command.State)
: IO CheckpointedParseResult := do
    -- IO.println s!"Extracting proof for segment between lines {prev} and {next}"
    let contextBeforeDecl := get_in_between_content fileContent prev next
    -- IO.println s!"Processing declaration at line {decl.startPos.line}-{decl.endPos.line}"
    -- IO.println s!"Declaration text:\n{decl.text}\n---"
    -- IO.println s!"--- Context Before Decl ----\n{contextBeforeDecl}\n--- Context Before Decl ----\n"
    let chkpt_parse_res ← parseTactics contextBeforeDecl none cmd_state
    return chkpt_parse_res

/-- Extract proofs from multiple declarations -/
unsafe def extractProofsFromDecls (decls : Array DeclInfo) (fileContent : String) : IO (Array DeclInfo) := do
  let mut result := #[]
  let mut prev := 0
  let mut cmd_state : Option Command.State := none
  let mut next := 0
  let mut extra_content : Option String := none
  for decl in decls do
    if is_proof_extraction_needed decl then
      next := decl.startPos.line - 1
      let chkpt_parse_res ← parse_between fileContent prev next cmd_state
      -- IO.println s!"--- Context Before Decl ----\n{contextBeforeDecl}\n--- Context Before Decl ----\n"
      let parse_res := chkpt_parse_res.parseResult
      if parse_res.errors.size > 0 then
        -- supply the extra content to compile from
        extra_content := get_in_between_content fileContent prev next
        -- IO.println s!"Re-parsing declaration at lines: \n{extra_content.get!}"
        -- IO.println s!"\nDeclaration text:\n{decl.text}\n---"
        -- DO NOT update cmd_state yet
      else
        cmd_state := chkpt_parse_res.chkptState
        extra_content := none
        prev := next + 1
      let cmd_st ← match cmd_state with
        | some st => pure st
        | none => panic! "Failed to get valid cmd_state before processing declaration"
      let processed ← extractProofFromDecl decl cmd_st extra_content
      result := result.push processed
    else
      result := result.push decl
  return result

end TacticParser
