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
  position : Nat  -- UTF-8 byte position in the text
  delimiterType : String  -- ":=", "where", or "|"
  deriving Repr, BEq

instance : Ord DelimiterCandidate where
  compare a b := compare a.position b.position

/-- Try to parse a text snippet and return true if it parses without errors -/
def tryParseSuccessfully (text : String) (cmdState : Option Command.State): IO Bool := do
  let chkpt_result ← parseTactics text none cmdState
  let parse_res := chkpt_result.1
  let new_cmd_state := chkpt_result.2
  pure (parse_res.errors.size == 0 ∧ new_cmd_state.isSome)

private def isIdentBytePE (b : UInt8) : Bool :=
  (b >= 97 && b <= 122) ||  -- a-z
  (b >= 65 && b <= 90)  ||  -- A-Z
  (b >= 48 && b <= 57)  ||  -- 0-9
  b == 95 ||                -- _
  b == 39 ||                -- '
  b >= 128                  -- non-ASCII UTF-8 bytes

/-- Find all top-level candidate proof delimiter positions (`:=`, `where`, `|`)
    in strictly ascending UTF-8 byte offset order.
    Skips line comments (`--`), nested block comments (`/- ... -/`), string
    literals (`"..."`), char literals (`'c'` without confusing identifier primes
    like `foo'`), and anything nested inside `(...)`, `[...]`, `{...}`, or `⟨...⟩`.
    Enforces word boundaries on `where` (ignoring `everywhere`, etc.) and
    operator boundaries on `|` (ignoring `||`, `<|`, `|>`, `|]`, `[|`, `|}`). -/
def findCandidateDelimiters (text : String) : List DelimiterCandidate := Id.run do
  let bytes := text.toUTF8
  let n := bytes.size
  let mut candidates : Array DelimiterCandidate := #[]
  let mut i : Nat := 0
  let mut inLineComment : Bool := false
  let mut blockDepth : Nat := 0
  let mut inString : Bool := false
  let mut escaped : Bool := false
  let mut inChar : Bool := false
  let mut charEscaped : Bool := false
  let mut parenDepth : Nat := 0
  let mut bracketDepth : Nat := 0
  let mut braceDepth : Nat := 0
  let mut angleDepth : Nat := 0
  let whereBytes := "where".toUTF8

  while i < n do
    let b := bytes[i]!
    if inLineComment then
      if b == 10 then
        inLineComment := false
      i := i + 1
    else if blockDepth > 0 then
      if i + 1 < n && b == 47 && bytes[i + 1]! == 45 then  -- /-
        blockDepth := blockDepth + 1
        i := i + 2
      else if i + 1 < n && b == 45 && bytes[i + 1]! == 47 then  -- -/
        blockDepth := blockDepth - 1
        i := i + 2
      else
        i := i + 1
    else if inString then
      if escaped then
        escaped := false
      else if b == 92 then  -- \
        escaped := true
      else if b == 34 then  -- "
        inString := false
      i := i + 1
    else if inChar then
      if charEscaped then
        charEscaped := false
      else if b == 92 then
        charEscaped := true
      else if b == 39 || b == 10 then  -- ' or newline safety exit
        inChar := false
      i := i + 1
    else
      -- Comments
      if i + 1 < n && b == 45 && bytes[i + 1]! == 45 then  -- --
        inLineComment := true
        i := i + 2
      else if i + 1 < n && b == 47 && bytes[i + 1]! == 45 then  -- /-
        blockDepth := 1
        i := i + 2
      -- String literal
      else if b == 34 then  -- "
        inString := true
        i := i + 1
      -- Char literal (ONLY when not preceded by an identifier byte, so `foo'` is untouched)
      else if b == 39 then  -- '
        let prevIsIdent := i > 0 && isIdentBytePE bytes[i - 1]!
        if !prevIsIdent && i + 2 < n &&
           (bytes[i + 2]! == 39 || (bytes[i + 1]! == 92 && i + 3 < n && bytes[i + 3]! == 39)) then
          inChar := true
        i := i + 1
      -- Nesting delimiters
      else if b == 40 then  -- (
        parenDepth := parenDepth + 1
        i := i + 1
      else if b == 41 then  -- )
        if parenDepth > 0 then parenDepth := parenDepth - 1
        i := i + 1
      else if b == 91 then  -- [
        bracketDepth := bracketDepth + 1
        i := i + 1
      else if b == 93 then  -- ]
        if bracketDepth > 0 then bracketDepth := bracketDepth - 1
        i := i + 1
      else if b == 123 then  -- {
        braceDepth := braceDepth + 1
        i := i + 1
      else if b == 125 then  -- }
        if braceDepth > 0 then braceDepth := braceDepth - 1
        i := i + 1
      -- UTF-8 `⟨` = 0xE2 0x9F 0xA8, `⟩` = 0xE2 0x9F 0xA9
      else if i + 2 < n && b == 0xE2 && bytes[i + 1]! == 0x9F && bytes[i + 2]! == 0xA8 then
        angleDepth := angleDepth + 1
        i := i + 3
      else if i + 2 < n && b == 0xE2 && bytes[i + 1]! == 0x9F && bytes[i + 2]! == 0xA9 then
        if angleDepth > 0 then angleDepth := angleDepth - 1
        i := i + 3
      else
        let atTop := parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 && angleDepth == 0
        if atTop && b == 58 && i + 1 < n && bytes[i + 1]! == 61 then  -- :=
          let prevOk := i == 0 || bytes[i - 1]! != 58  -- not ::=
          let nextOk := i + 2 >= n || bytes[i + 2]! != 61  -- not :==
          if prevOk && nextOk then
            candidates := candidates.push { position := i, delimiterType := ":=" }
          i := i + 2
        else if atTop && b == 119 && i + 5 <= n &&  -- "where"
                bytes[i + 1]! == whereBytes[1]! &&
                bytes[i + 2]! == whereBytes[2]! &&
                bytes[i + 3]! == whereBytes[3]! &&
                bytes[i + 4]! == whereBytes[4]! then
          let prevOk := i == 0 || !isIdentBytePE bytes[i - 1]!
          let nextOk := i + 5 >= n || !isIdentBytePE bytes[i + 5]!
          if prevOk && nextOk then
            candidates := candidates.push { position := i, delimiterType := "where" }
            i := i + 5
          else
            i := i + 1
        else if atTop && b == 124 then  -- |
          let prevOk := i == 0 ||
            (bytes[i - 1]! != 124 && bytes[i - 1]! != 60 && bytes[i - 1]! != 91 && bytes[i - 1]! != 123)
          let nextOk := i + 1 >= n ||
            (bytes[i + 1]! != 124 && bytes[i + 1]! != 62 && bytes[i + 1]! != 45 && bytes[i + 1]! != 93 && bytes[i + 1]! != 125)
          if prevOk && nextOk then
            candidates := candidates.push { position := i, delimiterType := "|" }
          i := i + 1
        else
          i := i + 1

  return candidates.toList

def is_proof_extraction_needed (declInfo : DeclInfo) : Bool :=
  declInfo.declType == DeclType.theorem ||
  declInfo.declType == DeclType.lemma ||
  declInfo.declType == DeclType.example

def get_content_before_decl (fileContent : String) (declLineNum : Nat) : String :=
  let lines := fileContent.splitOn "\n"
  let beforeLines := lines.take (declLineNum - 1)
  (String.intercalate "\n" beforeLines) ++ "\n"

def get_in_between_content (fileContent : String) (startLine : Nat) (endLine : Nat) : String :=
  if endLine < startLine || endLine == 0 then
    ""
  else
    let lines := fileContent.splitOn "\n"
    let startIdx := if startLine == 0 then 0 else startLine - 1
    let betweenLines := (lines.take endLine).drop startIdx
    String.intercalate "\n" betweenLines

/-- Extract proof from a declaration by testing candidate delimiters in ascending
    position order (so e.g. `theorem ... : C where field := by ...` splits at
    the outer `where`, not the inner `:=`). -/
unsafe def extractProofFromDecl
  (declInfo : DeclInfo)
  (cmdState : Option Command.State)
  (extra_content: Option String := none) : IO DeclInfo := do
  -- Only process theorem, lemma, example
  if !is_proof_extraction_needed declInfo then
    panic! s!"extractProofFromDecl called on non-proof decl: {declInfo.declType}"

  let text := declInfo.text

  -- Find all top-level candidate delimiters in ascending byte-offset order
  let candidates := findCandidateDelimiters text

  -- Test each candidate
  for candidate in candidates do
    let splitPos : String.Pos.Raw := ⟨candidate.position⟩
    let beforeDelimiter := String.Pos.Raw.extract text ⟨0⟩ splitPos

    -- Build test text with full context
    let mut statementOnly := match candidate.delimiterType with
      | "|" => beforeDelimiter.trim ++ " := sorry"
      | ":=" => beforeDelimiter ++ " := sorry"
      | "where" => beforeDelimiter ++ " := sorry"
      | _ => beforeDelimiter ++ " := sorry"

    -- If extra content is provided, prepend it
    statementOnly :=
      match extra_content with
      | some extra =>
        if extra.isEmpty then statementOnly
        else extra ++ (if extra.endsWith "\n" then statementOnly else "\n" ++ statementOnly)
      | none => statementOnly

    -- Try to parse/elaborate the stubbed statement
    let success ← tryParseSuccessfully statementOnly cmdState
    if success then
      -- Found valid split!
      let proof := String.Pos.Raw.extract text splitPos text.rawEndPos
      let thrm := beforeDelimiter
      return { declInfo with proof := some proof.trim, text := thrm.trim }

  -- No valid split found - no proof
  return { declInfo with proof := none }

unsafe def parse_between
(fileContent : String)
(prev: Nat)
(next: Nat)
(cmd_state : Option Command.State)
: IO CheckpointedParseResult := do
    let contextBeforeDecl := get_in_between_content fileContent prev next
    if contextBeforeDecl.trim.isEmpty && cmd_state.isSome then
      return {
        parseResult := { trees := #[], errors := #[] }
        chkptState := cmd_state
        lineNum := some 0
      }
    let chkpt_parse_res ← parseTactics contextBeforeDecl none cmd_state
    return chkpt_parse_res

/-- Extract proofs from multiple declarations.
    Optionally accepts a pre-initialized `initialCmdState` (from `processHeader`)
    and `initialPrevLine` (1-indexed first line after the header) to avoid
    re-importing the file's header modules. -/
unsafe def extractProofsFromDecls
  (decls : Array DeclInfo)
  (fileContent : String)
  (initialCmdState : Option Command.State := none)
  (initialPrevLine : Nat := 0)
  : IO (Array DeclInfo) := do
  let mut result := #[]
  let mut prev := initialPrevLine
  let mut cmd_state : Option Command.State := initialCmdState
  let mut next := 0
  let mut extra_content : Option String := none
  for decl in decls do
    if is_proof_extraction_needed decl then
      next := if decl.startPos.line > 0 then decl.startPos.line - 1 else 0
      let chkpt_parse_res ← parse_between fileContent prev next cmd_state
      let parse_res := chkpt_parse_res.parseResult
      if parse_res.errors.size > 0 ∨ chkpt_parse_res.chkptState.isNone then
        -- supply the extra content to compile from
        let extra := get_in_between_content fileContent prev next
        extra_content := if extra.trim.isEmpty then none else some extra
      else
        cmd_state := chkpt_parse_res.chkptState
        extra_content := none
        prev := next + 1
      let processed ← extractProofFromDecl decl cmd_state extra_content
      result := result.push processed
    else
      result := result.push decl
  return result

end TacticParser
