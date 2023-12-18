import data.real.basic
import data.nat.factorial.basic


theorem mod_arith_2_eq_zero_or_one
(x : ℕ) : x % 2 = 0 ∨ x % 2 = 1 :=
begin
  induction x,
  {simp,},
  begin
    rw nat.succ_eq_add_one,
    rw nat.add_mod,
    cases x_ih,
    {
      right,
      rw x_ih,
      simp
    },
    {
      left,
      rw x_ih,
      simp
    }
  end
end

theorem n_less_2_pow_n
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=
begin
  induction h₀ with k h₀ IH,
  { norm_num },
  {
    calc k + 1 < 2 * 2^k : by linarith
  }
end