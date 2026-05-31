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
