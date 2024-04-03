import data.real.basic
import data.nat.factorial.basic

theorem bool_iff : ∀ (f : bool → bool), 
f = (λ x, x) ∨ 
f = (λ x, !x) ∨ 
f = (λ x, ff) ∨ 
f = (λ x, tt) :=
begin

end