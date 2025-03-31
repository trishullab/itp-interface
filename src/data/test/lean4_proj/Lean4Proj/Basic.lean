import Mathlib
namespace Lean4Proj1

def hello := "world"

theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp


theorem test2 : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩


-- show a proof which uses calc
theorem test_calc (n: Nat) : n^2 + 2*n + 1 = (n + 1)*(n + 1) := by
calc
    _ = n^2 + n*2 + 1 := by rw [Nat.mul_comm 2 n]
    _ = n^2 + (n + n) + 1 := by rw [Nat.mul_two]
    _ = n^2 + n + n + 1 := by rw [←Nat.add_assoc]
    _ = n*n + n + n + 1 := by rw [Nat.pow_two]
    _ = n*n + n*1 + n + 1 := by rw [Nat.mul_one n]
    _ = n*(n + 1) + n + 1 := by rw [Nat.left_distrib n n 1]
    _ = n*(n + 1) + (n + 1) := by rw [Nat.add_assoc]
    _ = n*(n + 1) + 1*(n + 1) := by rw (config := { occs := .pos [2]}) [←Nat.mul_one (n + 1), Nat.mul_comm]
    _ = (n + 1)*(n + 1) := by rw [Nat.right_distrib n 1 (n + 1)]
done

end Lean4Proj1

namespace Lean4Proj2

example : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩

theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp
done

theorem test3 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp

theorem imo_1959_p1
  (n : ℕ)
  (h₀ : 0 < n) :
  Nat.gcd (21*n + 4) (14*n + 3) = 1 := by
rw [Nat.gcd_rec]
rw [Nat.mod_eq_of_lt (by linarith)]
rw [Nat.gcd_rec]
rw [Nat.gcd_rec]
have eq₂ : (21 * n + 4) % (14 * n + 3) = 7 * n + 1 := by
    have eq₁ : 21 * n + 4 = (14 * n + 3) + (7 * n + 1) := by ring
    rw [eq₁, Nat.add_mod, Nat.mod_self, zero_add]
    have h₂ : 7 * n + 1 < 14 * n + 3 := by linarith
    rw [Nat.mod_eq_of_lt]
    rw [Nat.mod_eq_of_lt]
    exact h₂
    rw [Nat.mod_eq_of_lt]
    exact h₂
    exact h₂
rw [eq₂]
sorry


end Lean4Proj2
