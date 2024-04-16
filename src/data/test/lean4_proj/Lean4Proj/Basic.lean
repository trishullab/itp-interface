def hello := "world"

theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp


theorem test2 : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩

example : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩

theorem test1 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp

theorem test3 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp
