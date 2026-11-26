import Lean
import Lean.Data.Json
import TacticParser.Types
namespace TacticParser

open Lean
open Lean.Parser

/-- Recursively unwrap `... in <cmd>` combinators (e.g., `set_option ... in`,
    `variable ... in`, `open ... in`, `omit ... in`, `include ... in`,
    `attribute ... in`) to reach the inner command syntax node. -/
partial def unwrapCommandIn (stx : Syntax) : Syntax :=
  if stx.getKind == `Lean.Parser.Command.in && stx.getNumArgs >= 3 then
    unwrapCommandIn (stx.getArg (stx.getNumArgs - 1))
  else
    stx

/-- Identify the type of declaration from syntax -/
partial def identifySomeDeclType (stx : Syntax) : Option (DeclType × Nat) :=
  let stx := unwrapCommandIn stx
  let kind := stx.getKind
  -- Check if this is a declaration wrapper, if so, look inside
  if kind == `Lean.Parser.Command.declaration then
    match stx with
    | Syntax.node _ _ args =>
      -- Look for the actual declaration type in the children
      -- NOTE: using findIdx? for backward compatibility to Lean 4.15
      let idx := (args.findIdx? (fun a => (identifySomeDeclType a).isSome)).getD args.size
      if idx = args.size then
        some (.unknown, idx)
      else
        let decl := args[idx]!
        match identifySomeDeclType decl with
        | some dt => some (dt.1, idx)
        | none => some (.unknown, idx)
    | _ => none
  else if kind == `Lean.Parser.Command.end then some (.end, 0)
  else if kind == `Lean.Parser.Command.namespace then some (.namespace, 0)
  else if kind == `Lean.Parser.Command.section then some (.section, 0)
  else if kind == `Lean.Parser.Command.mutual then some (.mutual, 0)
  else if kind == `Lean.Parser.Command.inductive then some (.inductive, 0)
  else if kind == `Lean.Parser.Command.coinductive then some (.inductive, 0)
  else if kind == `Lean.Parser.Command.theorem then some (.theorem, 0)
  else if kind == `Lean.Parser.Command.definition then some (.def, 0)
  else if kind == `Lean.Parser.Command.abbrev then some (.abbrev, 0)
  else if kind == `Lean.Parser.Command.opaque then some (.opaque, 0)
  else if kind == `Lean.Parser.Command.axiom then some (.axiom, 0)
  else if kind == `Lean.Parser.Command.structure then some (.structure, 0)
  else if kind == `Lean.Parser.Command.classDecl then some (.class_decl, 0)
  else if kind == `Lean.Parser.Command.classInductive then some (.class_decl, 0)
  else if kind == `Lean.Parser.Command.instance then some (.instance, 0)
  else if kind == `Lean.Parser.Command.example then some (.example, 0)
  else if kind == `Lean.Parser.Command.otherDecl then some (.other, 0)
  else if kind == `lemma || kind.toString.endsWith ".lemma" then some (.lemma, 0)
  else none

/-- Identify the type of declaration from syntax -/
def identifyDeclType (stx : Syntax) : DeclType :=
  match identifySomeDeclType stx with
  | some dt => dt.1
  | none => .unknown

/-- Check if a syntax node is an attribute/modifier or body subtree that should
    not be searched when locating a declaration's own identifier. -/
def isModifierOrAttribute (stx : Syntax) : Bool :=
  let kind := stx.getKind
  let ks := kind.toString
  kind == `Lean.Parser.Term.attrInstance ||
  kind == `Lean.Parser.Command.docComment ||
  kind == `Lean.Parser.Term.attributes ||
  kind == `Lean.Parser.Command.visibility ||
  kind == `Lean.Parser.Command.declSig ||
  kind == `Lean.Parser.Command.optDeclSig ||
  kind == `Lean.Parser.Command.declVal ||
  kind == `Lean.Parser.Command.declValSimple ||
  kind == `Lean.Parser.Command.declValEqns ||
  kind == `Lean.Parser.Command.whereDecls ||
  kind == `Lean.Parser.Command.structFields ||
  kind == `Lean.Parser.Command.ctor ||
  ks.startsWith "Lean.Parser.Command.declModifiers" ||
  ks.startsWith "Lean.Parser.Command.termination"

/-- Locate the `Lean.Parser.Command.declId` node of the outer declaration and
    return its identifier string. Does not descend into modifiers, signatures,
    or bodies. -/
partial def findDeclId? (stx : Syntax) : Option String :=
  let kind := stx.getKind
  if kind == `Lean.Parser.Command.declId then
    if stx.getNumArgs > 0 then
      match stx.getArg 0 with
      | Syntax.ident _ _ name _ => some name.toString
      | other => findDeclId? other
    else
      none
  else if isModifierOrAttribute stx then
    none
  else
    match stx with
    | Syntax.node _ _ args =>
      args.findSome? findDeclId?
    | _ => none

/-- Extract the identifier from a `namespace`, `section`, or `end` command,
    returning `none` for unnamed `section` or `end`. -/
partial def findScopeIdent? (stx : Syntax) : Option String :=
  match stx with
  | Syntax.ident _ _ name _ => some name.toString
  | Syntax.node _ kind args =>
    if isModifierOrAttribute stx || kind == `Lean.Parser.Command.visibility then
      none
    else
      args.findSome? findScopeIdent?
  | _ => none

/-- Extract the name of the declaration from syntax tree -/
partial def extractDeclName (stx : Syntax) : String :=
  let inner := unwrapCommandIn stx
  let kind := inner.getKind
  if kind == `Lean.Parser.Command.namespace ||
     kind == `Lean.Parser.Command.section ||
     kind == `Lean.Parser.Command.end then
    (findScopeIdent? inner).getD Name.anonymous.toString
  else if kind == `Lean.Parser.Command.mutual then
    Name.anonymous.toString
  else
    -- Prefer the authoritative `declId` syntax node (handles dotted names,
    -- French-quoted identifiers, `abbrev`, `opaque`, `public`, `meta`, `lemma`,
    -- named `instance`, `structure`, `inductive`, `class`, `axiom`, `def`, `theorem`).
    match findDeclId? inner with
    | some name => name
    | none =>
      -- Unnamed `instance`, `example`, or unrecognized command.
      match identifyDeclType inner with
      | .instance | .example | .mutual => Name.anonymous.toString
      | _ =>
        -- Fallback for custom macro declarations without a `declId` wrapper
        match inner with
        | Syntax.ident _ _ name _ => name.toString
        | Syntax.node _ _ args =>
          (args.findSome? fun arg =>
            if isModifierOrAttribute arg then
              none
            else
              match arg with
              | Syntax.ident _ _ name _ =>
                let s := name.toString
                if s != Name.anonymous.toString &&
                   s != "theorem" && s != "def" && s != "lemma" &&
                   s != "abbrev" && s != "opaque" &&
                   s != "inductive" && s != "coinductive" &&
                   s != "structure" && s != "class" &&
                   s != "instance" && s != "axiom" &&
                   s != "public" && s != "private" && s != "protected" &&
                   s != "noncomputable" && s != "partial" && s != "unsafe" &&
                   s != "meta" then
                  some s
                else
                  none
              | _ => none
          ).getD Name.anonymous.toString
        | _ => Name.anonymous.toString

/-- Find the byte range `(start, stop)` of the outer declaration's own
    `/-- ... -/` docComment node (if any), unwrapping `... in` combinators
    and NEVER descending into structure fields, inductive constructors, or
    `where` helper definitions. -/
partial def findOuterDocCommentRange? (stx : Syntax) : Option (String.Pos.Raw × String.Pos.Raw) :=
  let inner := unwrapCommandIn stx
  let rec findInModifiers (n : Syntax) : Option (String.Pos.Raw × String.Pos.Raw) :=
    let k := n.getKind
    if k == `Lean.Parser.Command.docComment then
      match n.getPos?, n.getTailPos? with
      | some s, some e => some (s, e)
      | _, _ => none
    else if k == `Lean.Parser.Command.declaration ||
            k.toString.startsWith "Lean.Parser.Command.declModifiers" ||
            k == `lemma || k.toString.endsWith ".lemma" ||
            k == `Lean.Parser.Command.structure ||
            k == `Lean.Parser.Command.classDecl ||
            k == `Lean.Parser.Command.inductive ||
            k == `Lean.Parser.Command.coinductive ||
            k == `Lean.Parser.Command.classInductive ||
            k == `Lean.Parser.Command.theorem ||
            k == `Lean.Parser.Command.definition ||
            k == `Lean.Parser.Command.abbrev ||
            k == `Lean.Parser.Command.opaque ||
            k == `Lean.Parser.Command.axiom ||
            k == `Lean.Parser.Command.instance ||
            k == `Lean.Parser.Command.example ||
            k == `null then
      match n with
      | Syntax.node _ _ args => args.findSome? findInModifiers
      | _ => none
    else
      none
  findInModifiers inner

/-- Helper: check if a byte is an ASCII identifier continuation character. -/
private def isIdentByte (b : UInt8) : Bool :=
  (b >= 97 && b <= 122) ||  -- a-z
  (b >= 65 && b <= 90)  ||  -- A-Z
  (b >= 48 && b <= 57)  ||  -- 0-9
  b == 95 ||                -- _
  b == 39 ||                -- '
  b == 46 ||                -- .
  b == 63 ||                -- ?
  b == 33 ||                -- !
  b >= 128                  -- non-ASCII UTF-8 bytes

private def isSpaceByte (b : UInt8) : Bool :=
  b == 32 || b == 9 || b == 10 || b == 13

/-- Rewrite standalone `lemma` command keywords to `theorem` in a throwaway copy
    used ONLY when the parser environment does not register the `lemma` macro
    (e.g., `Init`-only standalone parsing without Mathlib/Batteries).
    Returns `(rewrittenContent, origEditByteOffsets)` where each entry in
    `origEditByteOffsets` is the original byte offset where 5-byte `"lemma"` was
    replaced with 7-byte `"theorem"`. Skips line comments, nested block comments,
    and string literals; never mutates the original content. -/
def rewriteLemmaForParse (content : String) : String × Array Nat := Id.run do
  let bytes := content.toUTF8
  let n := bytes.size
  let mut out : ByteArray := ByteArray.emptyWithCapacity (n + 64)
  let mut edits : Array Nat := #[]
  let mut i : Nat := 0
  let mut blockDepth : Nat := 0
  let mut inLineComment : Bool := false
  let mut inString : Bool := false
  let mut escaped : Bool := false
  -- Track whether we are at a plausible command/declaration-keyword position
  -- (start of line, or preceded by whitespace / `]` after attributes / modifiers).
  let lemmaBytes : ByteArray := "lemma".toUTF8
  let theoremBytes : ByteArray := "theorem".toUTF8
  while i < n do
    let b := bytes[i]!
    if inLineComment then
      out := out.push b
      if b == 10 then
        inLineComment := false
      i := i + 1
    else if blockDepth > 0 then
      if i + 1 < n && b == 47 && bytes[i + 1]! == 45 then  -- /-
        out := out.push b |>.push bytes[i + 1]!
        blockDepth := blockDepth + 1
        i := i + 2
      else if i + 1 < n && b == 45 && bytes[i + 1]! == 47 then  -- -/
        out := out.push b |>.push bytes[i + 1]!
        blockDepth := blockDepth - 1
        i := i + 2
      else
        out := out.push b
        i := i + 1
    else if inString then
      out := out.push b
      if escaped then
        escaped := false
      else if b == 92 then  -- \
        escaped := true
      else if b == 34 then  -- "
        inString := false
      i := i + 1
    else
      -- Normal code
      if i + 1 < n && b == 45 && bytes[i + 1]! == 45 then  -- --
        out := out.push b |>.push bytes[i + 1]!
        inLineComment := true
        i := i + 2
      else if i + 1 < n && b == 47 && bytes[i + 1]! == 45 then  -- /-
        out := out.push b |>.push bytes[i + 1]!
        blockDepth := 1
        i := i + 2
      else if b == 34 then  -- "
        out := out.push b
        inString := true
        i := i + 1
      else if b == 108 && i + 5 <= n &&  -- 'l'
              bytes[i + 1]! == lemmaBytes[1]! &&
              bytes[i + 2]! == lemmaBytes[2]! &&
              bytes[i + 3]! == lemmaBytes[3]! &&
              bytes[i + 4]! == lemmaBytes[4]! then
        let prevOk := i == 0 || !isIdentByte bytes[i - 1]!
        let nextOk := i + 5 < n && isSpaceByte bytes[i + 5]!
        -- Only rewrite when `lemma` is at command/modifier position on its line
        -- (start of line or preceded only by spaces / modifiers / attributes / `in`).
        let lineOk := Id.run do
          if !prevOk || !nextOk then
            return false
          -- Scan backward to start of line to verify we aren't inside an expression
          -- (e.g., `foo lemma` or `:= lemma`).
          let mut j := i
          while j > 0 && bytes[j - 1]! != 10 do
            j := j - 1
          let prefixStr := (String.Pos.Raw.extract content ⟨j⟩ ⟨i⟩).trim
          if prefixStr.isEmpty then
            return true
          -- Allow modifiers, attributes (`]`), or `... in` before `lemma`.
          let lastTok := (prefixStr.splitOn " ").getLast!
          return lastTok == "public" || lastTok == "private" || lastTok == "protected" ||
                 lastTok == "noncomputable" || lastTok == "partial" || lastTok == "unsafe" ||
                 lastTok == "meta" || lastTok == "in" || lastTok.endsWith "]" ||
                 lastTok.endsWith "-/"
        if lineOk then
          edits := edits.push i
          out := out ++ theoremBytes
          i := i + 5
        else
          out := out.push b
          i := i + 1
      else
        out := out.push b
        i := i + 1
  return (String.fromUTF8! out, edits)

/-- Map a byte position in `parseContent` back to the exact byte position in
    `originalContent`, given sorted original byte offsets where `"lemma"` (5 bytes)
    was replaced by `"theorem"` (7 bytes). -/
def mapParsePosToOrig (edits : Array Nat) (parsePos : Nat) : Nat := Id.run do
  let mut shift : Nat := 0
  for origOff in edits do
    let editStartInParse := origOff + shift
    if parsePos < editStartInParse then
      break
    else if parsePos < editStartInParse + 7 then
      return origOff
    else
      shift := shift + 2
  return parsePos - shift

/-- Check if any rewritten `lemma` edit occurred in `[origStart, origEnd)`. -/
def hasLemmaEditInRange (edits : Array Nat) (origStart origEnd : Nat) : Bool :=
  edits.any fun o => o >= origStart && o < origEnd

/-- Check whether `env` already defines a `lemma` command syntax/macro
    (e.g., Mathlib or Batteries is imported). -/
def envSupportsLemma (env : Environment) : Bool :=
  env.contains `Batteries.Tactic.lemma ||
  env.contains `Mathlib.Tactic.lemma ||
  env.contains `Lean.Parser.Command.lemma

/-- Scope frame on the namespace/section stack: `(isNamespace, nameComponents)`. -/
private abbrev ScopeFrame := Bool × Array String

private def splitNonEmpty (s : String) : Array String :=
  if s.isEmpty || s == Name.anonymous.toString then
    #[]
  else
    ((s.splitOn ".").filter (fun p => !p.isEmpty)).toArray

/-- Pop/trim scope frames according to Lean 4's `end` semantics. -/
private def popScope (stack : Array ScopeFrame) (endName : String) : Array ScopeFrame := Id.run do
  if stack.isEmpty then
    return stack
  let endParts := splitNonEmpty endName
  if endParts.isEmpty then
    -- Unnamed `end`: close the innermost open scope (namespace or section).
    return stack.pop
  -- Named `end A.B`: match trailing components from the top of the stack.
  let mut st := stack
  let mut rem := endParts
  while !rem.isEmpty && !st.isEmpty do
    let (isNs, parts) := st.back!
    if parts.isEmpty then
      -- Unnamed section cannot match a named `end`; abort without corrupting stack.
      return stack
    if parts.size <= rem.size then
      -- Check if `parts` matches the suffix of `rem`.
      let offset := rem.size - parts.size
      let mut ok := true
      for k in [:parts.size] do
        if rem[offset + k]! != parts[k]! then
          ok := false
      if ok then
        st := st.pop
        rem := rem.extract 0 offset
      else
        return stack
    else
      -- Frame has more parts than `rem` (e.g. `namespace A.B` closed by `end B`).
      let offset := parts.size - rem.size
      let mut ok := true
      for k in [:rem.size] do
        if parts[offset + k]! != rem[k]! then
          ok := false
      if ok then
        st := st.pop.push (isNs, parts.extract 0 offset)
        rem := #[]
      else
        return stack
  if rem.isEmpty then st else stack

/-- Compute the active dotted namespace string from the scope stack. -/
private def currentNamespace (stack : Array ScopeFrame) : Option String :=
  let parts := stack.foldl (init := #[]) fun acc (isNs, ps) =>
    if isNs then acc ++ ps else acc
  if parts.isEmpty then none else some (String.intercalate "." parts.toList)

/-- Remove a `[docStart, docEnd)` byte slice from `[cmdStart, cmdEnd)` inside
    `originalContent`, cleaning up a single trailing newline/indentation gap
    so the remaining command text stays clean and valid Lean. -/
private def removeDocSpan (originalContent : String)
    (cmdStart cmdEnd docStart docEnd : String.Pos.Raw) : String :=
  let before := String.Pos.Raw.extract originalContent cmdStart docStart
  let after := String.Pos.Raw.extract originalContent docEnd cmdEnd
  let beforeTrim := before.trimRight
  let afterTrim := after.trimLeft
  if beforeTrim.isEmpty then
    afterTrim
  else if afterTrim.isEmpty then
    beforeTrim
  else
    beforeTrim ++ "\n" ++ afterTrim

/-- Collect namespace identifiers from an `open` / `open scoped` / `namespace`
    command so we can activate any scoped parser extensions (e.g., `Topology`,
    `Filter`, `BigOperators`, `NNReal`, `Pointwise`) they bring into scope. -/
private partial def collectScopeIdents (stx : Syntax) : Array Name :=
  match stx with
  | Syntax.ident _ _ name _ => #[name]
  | Syntax.node _ kind args =>
    if kind == `Lean.Parser.Command.openHiding ||
       kind == `Lean.Parser.Command.openOnly ||
       kind == `Lean.Parser.Command.openRenaming then
      if args.size > 0 then collectScopeIdents args[0]! else #[]
    else
      args.foldl (init := #[]) fun acc a => acc ++ collectScopeIdents a
  | _ => #[]

/-- Activate scoped syntax extensions for `open` / `open scoped` / `namespace` commands. -/
private def activateOpenScoped (env : Environment) (stx : Syntax) : Environment :=
  let inner := unwrapCommandIn stx
  let kind := inner.getKind
  if kind == `Lean.Parser.Command.open || kind == `Lean.Parser.Command.namespace then
    let names := collectScopeIdents inner
    names.foldl (init := env) fun e n =>
      Lean.Parser.parserExtension.activateScoped e n
  else
    env

/-- Core command-level declaration parser. Returns `(decls, parseErrorCount)`.
    Never mutates `originalContent`; `parseContent` may differ only by
    `lemma` -> `theorem` rewrites recorded in `lemmaEdits`. -/
def parseCommonWithStats
  (originalContent : String)
  (parseContent : String)
  (lemmaEdits : Array Nat)
  (parserState : ModuleParserState)
  (pmctx : ParserModuleContext)
  (inputCtx : InputContext)
  : IO (Array DeclInfo × Nat) := do
  -- First pass: parse all commands and track syntax error count, activating
  -- scoped notations whenever `open` / `open scoped` / `namespace` is seen.
  let mut commands : Array (String.Pos.Raw × Syntax) := #[]
  let mut pstate := parserState
  let mut curPmctx := pmctx
  let mut done := false
  let mut totalErrors : Nat := 0

  while !done do
    let startPos := pstate.pos
    let (stx, pstate', msgs) := parseCommand inputCtx curPmctx pstate {}
    pstate := pstate'
    totalErrors := totalErrors + msgs.toList.length

    if isTerminalCommand stx then
      done := true
    else
      curPmctx := { curPmctx with env := activateOpenScoped curPmctx.env stx }
      commands := commands.push (startPos, stx)
      if pstate.pos == startPos then
        done := true

  -- Second pass: extract declarations directly from the parsed ASTs
  let mut decls : Array DeclInfo := #[]
  let mut scopeStack : Array ScopeFrame := #[]

  for i in [:commands.size] do
    let (parsePos, stx) := commands[i]!
    let rawStart := stx.getPos?.getD parsePos
    let rawEnd := match stx.getTailPos? with
      | some tp => tp
      | none =>
        if i + 1 < commands.size then
          let (nextParsePos, nextStx) := commands[i + 1]!
          nextStx.getPos?.getD nextParsePos
        else
          parseContent.rawEndPos

    let origStart : String.Pos.Raw := ⟨mapParsePosToOrig lemmaEdits rawStart.byteIdx⟩
    let origEnd : String.Pos.Raw := ⟨mapParsePosToOrig lemmaEdits rawEnd.byteIdx⟩

    let declType := identifyDeclType stx
    let name := extractDeclName stx

    -- Update scope stack (`namespace`, `section`, `end`)
    if declType == DeclType.namespace then
      scopeStack := scopeStack.push (true, splitNonEmpty name)
    else if declType == DeclType.section then
      scopeStack := scopeStack.push (false, splitNonEmpty name)
    else if declType == DeclType.end then
      scopeStack := popScope scopeStack name

    -- Determine if this was originally a `lemma` rewritten to `theorem` for parsing
    let actualDeclType :=
      if declType == DeclType.theorem && hasLemmaEditInRange lemmaEdits origStart.byteIdx origEnd.byteIdx then
        DeclType.lemma
      else
        declType

    -- Extract ONLY the outer declaration's genuine `/-- ... -/` doc comment
    let (docString, cleanText) :=
      match findOuterDocCommentRange? stx with
      | some (docStartRaw, docEndRaw) =>
        let origDocStart : String.Pos.Raw := ⟨mapParsePosToOrig lemmaEdits docStartRaw.byteIdx⟩
        let origDocEnd : String.Pos.Raw := ⟨mapParsePosToOrig lemmaEdits docEndRaw.byteIdx⟩
        let docStr := (String.Pos.Raw.extract originalContent origDocStart origDocEnd).trim
        let cleaned := removeDocSpan originalContent origStart origEnd origDocStart origDocEnd
        (if docStr.isEmpty then none else some docStr, cleaned)
      | none =>
        let rawText := (String.Pos.Raw.extract originalContent origStart origEnd).trim
        (none, rawText)

    let namespc := currentNamespace scopeStack
    let start_pos := get_position_from_char_pos originalContent origStart.byteIdx
    let end_pos := get_position_from_char_pos originalContent origEnd.byteIdx

    let info : DeclInfo := {
      declType := actualDeclType
      name := name
      startPos := start_pos
      endPos := end_pos
      text := cleanText
      docString := docString
      namespc := namespc
    }
    decls := decls.push info

  return (decls, totalErrors)

/-- Backward-compatible wrapper returning only declarations. -/
def parseCommon
  (originalContent : String)
  (parserState : ModuleParserState)
  (pmctx : ParserModuleContext)
  (inputCtx : InputContext)
  : IO (Array DeclInfo) := do
  let (decls, _) ← parseCommonWithStats originalContent originalContent #[] parserState pmctx inputCtx
  return decls

/-- Load a default `Init` environment with parser extensions enabled.
    Requires `Lean.enableInitializersExecution` to have been called. -/
unsafe def loadInitEnv : IO Environment := do
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  try
    Lean.importModules #[{ module := `Init }] {} 0 (loadExts := true)
  catch _ =>
    Lean.importModules #[] {} 0

/-- Parse Lean 4 source using an explicit (optional) `Environment`, returning
    `(declarations, parseErrorCount)`. -/
unsafe def parseDeclsWithEnv
  (originalContent : String)
  (env? : Option Environment := none)
  (fileName : String := "<input>")
  : IO (Array DeclInfo × Nat) := do
  let env ← match env? with
    | some e => pure e
    | none => loadInitEnv
  let (parseContent, lemmaEdits) :=
    if envSupportsLemma env then
      (originalContent, #[])
    else
      rewriteLemmaForParse originalContent
  let inputCtx := mkInputContext parseContent fileName
  let (_, parserState, _) ← parseHeader inputCtx
  let pmctx : ParserModuleContext := { env := env, options := {} }
  parseCommonWithStats originalContent parseContent lemmaEdits parserState pmctx inputCtx

/-- Parse a Lean 4 source string and extract declaration information -/
unsafe def parseDecls (originalContent : String) (env? : Option Environment := none) : IO (Array DeclInfo) := do
  let (decls, _) ← parseDeclsWithEnv originalContent env? "<input>"
  return decls

/-- Parse a Lean 4 file and extract declaration information -/
unsafe def parseFile (filepath : System.FilePath) (env? : Option Environment := none) : IO (Array DeclInfo) := do
  let mut originalContent ← IO.FS.readFile filepath
  originalContent := if originalContent.endsWith "\n" then originalContent else originalContent ++ "\n"
  let (decls, _) ← parseDeclsWithEnv originalContent env? filepath.toString
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
    (info.text.take 50).copy ++ "\n ... more text ... \n" ++ (info.text.drop (info.text.length - 50)).copy
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

  IO.println s!"Parsing file: {filepath}"
  IO.println (String.mk (List.replicate 50 '='))

  if decls.isEmpty then
    IO.println "No declarations found."
  else
    IO.println s!"Found {decls.size} declaration(s):"
    IO.println ""
    for decl in decls do
      printDeclInfo decl

  match jsonOutput with
  | some outPath => exportToJson decls outPath
  | none => pure ()

end TacticParser
