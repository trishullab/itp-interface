import data.real.basic
import data.nat.factorial.basic
import data.nat.modeq


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


example : (∃ x: ℝ , 4 * x + 2 - 2 * x = 6) :=
begin
  use 2,
  ring_nf,
  -- let x : ℝ := _,
  -- use x,
  -- ring_nf,
  -- -- linear equation has a solution
  -- have h : 2
end

example : ∃ y, ∀ x: ℝ , 4 * x + 2 - 2 * x = 6 →  y = x :=
begin
  let x: ℝ := _,
  use x,
  intros y h,
  ring_nf at h,
  have h1 : 2 * y = 4,
  {
    linarith,
  },
  have h2 : y = 2,
  {
    linarith,
  },
  rw h2,
end



theorem mod_arith_2
(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
 intro h,
 rw nat.mul_mod,
 rw h,
 rw nat.zero_mul,
 refl,
end



theorem mod_arith_3 (x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
  intro h,
  rw nat.modeq.modeq_iff_dvd at *,
  obtain ⟨k, hk⟩ := h,
  rw hk,
  simp only [nat.mul_mod, mul_zero],
end