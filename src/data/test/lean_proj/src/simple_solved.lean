import data.real.basic
import data.nat.factorial.basic

theorem mul_add_left_distrib
(a b c : ℕ) : a * (b + c) = a * b + a * c :=
begin
  induction c,
  {
    simp,
  },
  {
    rw nat.mul_succ,
    rw ← nat.add_assoc (a * b) (a * c_n) a,
    rw ← c_ih,
    rw ← nat.mul_succ,
    rw nat.add_succ,
  }
end

theorem a_plus_b_a_minus_a 
(a b : ℕ) : a ≥ b → a*a - b*b = (a - b) * (a + b) :=
begin
  intro h,
  rw nat.mul_comm (a-b) (a+b),
  rw nat.mul_sub_left_distrib,
  rw nat.mul_comm (a+b) a,
  rw nat.mul_comm (a+b) b,
  rw mul_add_left_distrib,
  rw mul_add_left_distrib,
  rw nat.add_comm (a*a) (a*b),
  rw nat.mul_comm a b,
  rw nat.add_sub_add_left,
end

theorem mod_arith_2
(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
  intro h,
  rw nat.mul_mod,
  rw h,
  simp,
end

theorem mod_arith_2_eq_zero_or_one
(x : ℕ) : x % 2 = 0 ∨ x % 2 = 1 :=
begin
  induction x,
  {simp,},
  begin
    rw nat.succ_eq_add_one,
    rw nat.add_mod,
    cases x_ih,
    begin
      right,
      rw x_ih,
      simp
    end,
    {
      left,
      rw x_ih,
      simp
    }
  end
end


theorem sum_of_naturals_mod_2
(n : ℕ) : (n * (n + 1)) % 2 = 0 :=
begin
  cases mod_arith_2_eq_zero_or_one n,
  {
    rw nat.mul_mod,
    rw h,
    simp
  },
  rw nat.mul_mod,
  rw nat.add_mod,
  rw h,
  simp,
end

theorem ab_square: 
∀ (a b: ℝ), (a + b)^2 = a^2 + b^2 + 2*a*b :=
begin
  intros a b,
  rw pow_two,
  ring_nf,
end

theorem aime_1983_p2
(x p : ℝ)
(f : ℝ → ℝ)
(h₀ : 0 < p ∧ p < 15)
(h₁ : p ≤ x ∧ x ≤ 15)
(h₂ : f x = abs (x - p) + 
            abs (x - 15) + 
            abs (x - p - 15)) :
15 ≤ f x :=
begin
    have h₃ : f x ≥ 15, 
    {
      rw h₂,
      have h₄ : x - p ≥ 0, 
        by linarith [h₁.left],
      have h₅ : x - 15 ≤ 0, 
        by linarith [h₁.right],
      have h₆ : x - p - 15 ≤ 0, 
        by linarith [h₁.right, h₀.right],
      rw abs_of_nonneg h₄,
      rw abs_of_nonpos h₅,
      rw abs_of_nonpos h₆,
      linarith,
    },
    exact h₃,
end
