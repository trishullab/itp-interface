/-
Main executable: read base64 from stdin, output JSON to stdout.
Runs in a loop to avoid restart overhead.

The process should be started from the project directory for project-specific parsing.
-/
import TacticParser.Base64
import TacticParser.Types
import TacticParser.SyntaxWalker
import Lean

open Lean
open TacticParser

/-- Process a single request and output JSON -/
def processRequest (b64Input : String) (filePath : Option String := none) : IO Unit := do
  try
    -- Decode base64 to Lean code
    let leanCode ← match Base64.decode b64Input with
      | .ok code => pure code
      | .error msg =>
        -- Output error as JSON
        let result : ParseResult := { trees := #[], error := some s!"Base64 decode error: {msg}" }
        IO.println (toJson result).compress
        return

    -- Parse tactics from Lean code
    let result ← parseTactics leanCode filePath

    -- Output result as JSON
    IO.println (toJson result).compress

  catch e =>
    -- Output error as JSON
    let result : ParseResult := { trees := #[], error := some s!"Unexpected error: {e}" }
    IO.println (toJson result).compress

/-- Loop to process requests -/
partial def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) (filePath : Option String := none) : IO Unit := do
  -- Read input from stdin (base64)
  let line ← stdin.getLine
  let line := line.trim

  -- Exit on empty line or "exit" command
  if line.isEmpty || line = "exit" then
    return

  -- Process the request
  processRequest line filePath

  -- Flush output to ensure Python can read it
  stdout.flush

  -- Continue loop
  loop stdin stdout

/-- Start processing with filePath passed as an optional argument -/
def main (args : List String) : IO Unit := do
  let filePath : Option String :=
    match args with
    | [] => none
    | fp::_ => some fp

  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  loop stdin stdout filePath
