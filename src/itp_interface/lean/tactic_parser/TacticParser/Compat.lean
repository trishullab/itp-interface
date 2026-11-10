import Lean

/-!
# Compatibility shims for Lean version skew

In Lean v4.27, `String.Pos` was split into two types:

  * `String.Pos s` (new, dependent): `{ offset : Pos.Raw, isValid : offset.IsValid s }`
  * `String.Pos.Raw` (new, flat):    `{ byteIdx : Nat }` — the old `String.Pos`

Several `String` operations also changed return type:

  * `String.drop`, `String.take`, `String.extract`, `String.takeWhile`, etc.
    now return `String.Slice` (which has a `.copy : String.Slice → String`).
    Pre-v4.27 they returned `String` directly.

The `TacticParser` source uses the v4.27 spellings (`String.Pos.Raw.extract`,
`.offset.byteIdx`, `.copy`, `rawEndPos`). To stay buildable on the README's
documented range (4.15.0 – 4.24.0) as well as v4.27.0, this file provides
shims that are elaborated *only* on pre-v4.27 toolchains; on v4.27+ they are
skipped, so the real core definitions are used.

The `compat_pre_v427` macro is a tiny preprocessor that runs its body iff
`Lean.versionString` parses to `< 4.27`.
-/

open Lean Elab Command

/-- True iff the current Lean toolchain is older than v4.27. -/
private def itpInterface_isPreV427 : Bool := Id.run do
  let parts := Lean.versionString.splitOn "."
  let some majorStr := parts.head?         | return false
  let some minorStr := parts[1]?           | return false
  let some major    := majorStr.toNat?     | return false
  -- Strip any suffix like "-rc1" before parsing minor.
  let minorClean := minorStr.takeWhile Char.isDigit
  let some minor    := minorClean.toNat?   | return false
  return major < 4 || (major == 4 && minor < 27)

/-- Elaborate the body iff the current Lean toolchain is older than v4.27. -/
syntax (name := compatPreV427) "compat_pre_v427 " command* : command

@[command_elab compatPreV427]
def elabCompatPreV427 : CommandElab := fun stx => do
  if itpInterface_isPreV427 then
    let cmds := stx[1].getArgs
    for c in cmds do
      elabCommand c

-- ---------------------------------------------------------------------------
-- Removed-`String`-method shims.
--
-- The Lean 4 `String` type was reworked around UTF-8 slices (landing by v4.35)
-- and the whole-`String` spellings `String.trim`, `String.trimLeft`,
-- `String.trimRight`, `String.dropRightWhile`, `String.takeRightWhile`, and
-- `String.dropRight` were removed. The `TacticParser` source still uses those
-- spellings (`s.trim`, `line.trimLeft`, `s.dropRightWhile p`, …).
--
-- Rather than gate on a version number (fragile — the exact removal release is
-- easy to get wrong, and different methods could in principle go at different
-- times), each shim below is elaborated only when the corresponding `String.*`
-- constant is ABSENT from the current environment. On toolchains that still
-- provide the core definition the shim is skipped and the core version is used;
-- on toolchains where it has been removed the shim supplies it. The shims use
-- only primitives stable across the documented range (`String.toList`,
-- `String.push`, `Char.isWhitespace`, and `List` operations), so every call
-- site compiles unchanged on all supported toolchains.
-- ---------------------------------------------------------------------------

/-- Elaborate the brace-delimited body iff `name✝` is absent from the current
    environment. The single leading identifier is the fully-qualified constant to
    probe; the body is brace-delimited so its `command*` does not swallow the
    declarations that follow. -/
syntax (name := compatIfMissing) "compat_if_missing " ident "{" command* "}" : command

@[command_elab compatIfMissing]
def elabCompatIfMissing : CommandElab := fun stx => do
  let name := stx[1].getId
  unless (← getEnv).contains name do
    for c in stx[3].getArgs do
      elabCommand c

/-- Build a `String` from characters without `String.mk` (deprecated on recent
    toolchains) or `String.ofList` (absent on older ones). Always defined; only
    used by the shims below. -/
private def String.itpOfChars (cs : List Char) : String :=
  cs.foldl (fun s c => s.push c) ""

/- Shim for a removed `String.trimLeft`: drop leading whitespace. -/
compat_if_missing String.trimLeft {
  def String.trimLeft (s : String) : String :=
    String.itpOfChars (s.toList.dropWhile Char.isWhitespace)
}

/- Shim for a removed `String.trimRight`: drop trailing whitespace. -/
compat_if_missing String.trimRight {
  def String.trimRight (s : String) : String :=
    String.itpOfChars (s.toList.reverse.dropWhile Char.isWhitespace).reverse
}

/- Shim for a removed `String.trim`: drop leading and trailing whitespace.
    Uses `trimLeft`/`trimRight`, which resolve to the core or shimmed version. -/
compat_if_missing String.trim {
  def String.trim (s : String) : String :=
    s.trimLeft.trimRight
}

/- Shim for a removed `String.dropRightWhile`: drop the trailing run of
    characters satisfying `p`, returning the remaining prefix. -/
compat_if_missing String.dropRightWhile {
  def String.dropRightWhile (s : String) (p : Char → Bool) : String :=
    String.itpOfChars (s.toList.reverse.dropWhile p).reverse
}

/- Shim for a removed `String.takeRightWhile`: take the trailing run of
    characters satisfying `p`, in original order. -/
compat_if_missing String.takeRightWhile {
  def String.takeRightWhile (s : String) (p : Char → Bool) : String :=
    String.itpOfChars (s.toList.reverse.takeWhile p).reverse
}

/- Shim for a removed `String.dropRight`: drop the last `n` characters.
    Nat subtraction saturates at 0, so `n ≥ length` yields the empty string. -/
compat_if_missing String.dropRight {
  def String.dropRight (s : String) (n : Nat) : String :=
    let cs := s.toList
    String.itpOfChars (cs.take (cs.length - n))
}

-- ---------------------------------------------------------------------------
-- Pre-v4.27 shims.
--
-- On v4.27 these are NOT elaborated; the real core definitions are used.
-- On v4.15–4.24 these introduce the v4.27 spellings as thin wrappers.
-- ---------------------------------------------------------------------------

compat_pre_v427
  /-- Pre-v4.27 alias: the (then-flat) `String.Pos` plays the role of the new
      `String.Pos.Raw`. -/
  abbrev String.Pos.Raw : Type := String.Pos

  namespace String.Pos.Raw
    /-- Pre-v4.27 shim mirroring the v4.27 positional spelling
        `String.Pos.Raw.extract s b e`. -/
    def extract (s : String) (b e : String.Pos) : String := s.extract b e
  end String.Pos.Raw

  /-- Pre-v4.27 shim: in v4.27 a `String` exposes a flat past-the-end accessor
      `rawEndPos : Pos.Raw`. Pre-v4.27 only has `endPos`. -/
  def String.rawEndPos (s : String) : String.Pos := s.endPos

  namespace String.Pos
    /-- Pre-v4.27 shim: in v4.27 a dependent `String.Pos s` carries a `.offset : Pos.Raw`
        field. Pre-v4.27 the position is already flat, so `.offset` is the identity. -/
    def offset (p : String.Pos) : String.Pos := p
  end String.Pos

  /-- Pre-v4.27 shim: in v4.27 `String.drop`/`take`/`extract`/`takeWhile` return
      `String.Slice` and call sites use `.copy` to materialise back to `String`.
      Pre-v4.27 they already return `String`, so `.copy` is the identity. -/
  def String.copy (s : String) : String := s
