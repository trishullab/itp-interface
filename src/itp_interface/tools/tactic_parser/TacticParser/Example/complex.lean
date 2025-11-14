import TacticParser.Example.simple

namespace TacticParser.Example

theorem additive:
∀ {a b: Nat}, addNat a b = a + b := by
  intro a b
  induction a generalizing b
  simp [addNat]
  rename_i a ih
  simp [addNat, *]
  grind

theorem additive_identity1 :
∀ {a : Nat}, addNat 0 a = a := by
  simp [addNat]

theorem additive_identity2 :
∀ {a : Nat}, addNat a 0 = a := by
  simp [additive]

theorem additive_comm:
∀ {a b : Nat}, addNat a b = addNat b a := by
  simp [additive]
  grind
