import Lean
import TacticParser.DependencyParser
import TacticParser.ProofExtractor
import TacticParser.Types

open TacticParser


unsafe def analyze (filePath : String) : IO Unit := do
  let fileDepAnalysis ← analyzeFileDependencies filePath
  IO.println s!"Dependency Analysis for {filePath}:"
  let json := Lean.toJson fileDepAnalysis
  IO.println json.pretty

def testCodes : List String := [
  "TacticParser/Example/simple.lean",
  "TacticParser/Example/complex.lean"
]

unsafe def main : IO Unit := do
  IO.println "Testing User's Example"
  IO.println (String.mk (List.replicate 70 '='))

  for test in testCodes do
    let filePath := test
    analyze filePath
