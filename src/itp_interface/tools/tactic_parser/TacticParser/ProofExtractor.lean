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
unsafe def tryParseSuccessfully (text : String) (cmdState? : Option Command.State): IO Bool := do
  -- IO.println s!"Trying to parse text:\n{text}\n---"
  -- IO.println s!"With cmdState?: {cmdState?.isSome}"
  let (_, _, messages, _) ← try
    IO.processInput text cmdState? Options.empty
  catch e =>
    -- IO.println s!"Parsing exception: {e}"
    return false

  let mut errorInfos : Array ErrorInfo := #[]
  for msg in messages do
    if msg.severity == .error then
      let msgPos := Position.mk msg.pos.line msg.pos.column
      let errorInfo := ErrorInfo.mk (← msg.data.toString) msgPos
      errorInfos := errorInfos.push errorInfo
      -- IO.println s!"Parsing error: {← msg.data.toString} at {msg.pos}"
  if errorInfos.size > 0 then
    return false
  else
    return true

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

  let allCandidates := assignCandidates ++ whereCandidates ++ pipeCandidates
  allCandidates.toArray.qsort (fun a b => a.position < b.position) |>.toList

/-- Extract proof from a declaration by testing candidate delimiters -/
unsafe def extractProofFromDecl (declInfo : DeclInfo) (fileContent : String) : IO DeclInfo := do
  -- Only process theorem, lemma, example
  if declInfo.declType != DeclType.theorem &&
     declInfo.declType != DeclType.lemma &&
     declInfo.declType != DeclType.example then
    return declInfo

  let text := declInfo.text

  -- Extract context before this declaration (imports + previous declarations)
  -- Convert Position to byte offset
  let mut contextBeforeDecl :=
    let lines := fileContent.splitOn "\n"
    let beforeLines := lines.take (declInfo.startPos.line - 1)
    String.intercalate "\n" beforeLines
  contextBeforeDecl := contextBeforeDecl ++ "\n"

  -- IO.println s!"contextBeforeDecl:\n{contextBeforeDecl}\n---"

  let mut cmdState? : Option Command.State := none
  try
    let (initialCmdState, cmdState, messages, trees) ←  IO.processInput contextBeforeDecl none Options.empty
    cmdState? := some cmdState
  catch e =>
    cmdState? := none

  -- IO.println s!"cmdState?: {cmdState?.isSome}"

  -- Find all candidate delimiters
  let candidates := findCandidateDelimiters text

  -- IO.println s!"text:\n{text}\n---"

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
    let success ← tryParseSuccessfully statementOnly cmdState?
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

/-- Extract proofs from multiple declarations -/
unsafe def extractProofsFromDecls (decls : Array DeclInfo) (fileContent : String) : IO (Array DeclInfo) := do
  let mut result := #[]
  for decl in decls do
    let processed ← extractProofFromDecl decl fileContent
    result := result.push processed
  return result

end TacticParser
