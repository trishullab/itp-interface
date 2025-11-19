import TacticParser.SyntaxWalker
import Lean

open Lean
open TacticParser

/-- Print usage information -/
def printUsage : IO Unit := do
  IO.println "Usage: dependency_parser <lean_file_path> <json_output_path>"
  IO.println ""
  IO.println "Arguments:"
  IO.println "  <lean_file_path>     Path to the Lean file to analyze"
  IO.println "  <json_output_path>   Path where JSON output will be written"
  IO.println ""
  IO.println "Example:"
  IO.println "  lake env .lake/build/bin/syntax-walker MyFile.lean output.json"

unsafe def main (args : List String) : IO UInt32 := do
  match args with
  |
  [
     leanFilePath,
     jsonOutputPath
  ]
  =>
    try
      let filepath : System.FilePath := leanFilePath
      let jsonPath : System.FilePath := jsonOutputPath

      -- Check if input file exists
      if !(← filepath.pathExists) then
        IO.eprintln s!"Error: Input file not found: {filepath}"
        return 1

      let fileContent ← IO.FS.readFile filepath

      IO.println s!"Tactic Parsing file: {filepath}"

      -- Analyze the file and export to JSON
      let tactics ← parseTactics fileContent none none
      let tacticsJson := Lean.ToJson.toJson tactics.parseResult
      IO.FS.writeFile jsonPath tacticsJson.compress
      return 0
    catch e =>
      IO.eprintln s!"Error: {e}"
      return 1

  | _ =>
    IO.eprintln "Error: Invalid number of arguments"
    IO.eprintln ""
    printUsage
    return 1
