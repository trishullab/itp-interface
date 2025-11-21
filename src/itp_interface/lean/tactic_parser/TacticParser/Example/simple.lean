namespace TacticParser.Example

theorem test : ∀ {a : Nat}, a + 0 = a := by grind

theorem test1 : ∀ {a : Nat}, a + 0 = a
  | 0 => by simp
  | n + 1  => by simp

def addNat : Nat → Nat → Nat
  | 0, m => m
  | n + 1, m => addNat n (m + 1)

end TacticParser.Example
