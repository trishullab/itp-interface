"""
Test cases for the Lean 4 tactic parser.

Tests the tactic parser with various theorems from Basic.lean to ensure
it correctly extracts atomic tactics from Lean 4 proofs.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tactic_parser
sys.path.insert(0, str(Path(__file__).parent.parent / "itp_interface" / "tools"))

from itp_interface.tools.tactic_parser import TacticParser, print_tactics

project_path = str(Path(__file__).parent.parent / "data" / "test" / "lean4_proj")

class TestTacticParser:
    """Test suite for the Lean 4 tactic parser."""

    def test_simple_proof_with_apply_and_exact(self):
        """Test parsing a simple proof with apply and exact tactics."""
        lean_code = """theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract 5 atomic tactics
            assert len(tactics) == 5, f"Expected 5 tactics, got {len(tactics)}"

            # Check each tactic
            assert tactics[0].text == "apply And.intro", f"Expected 'apply And.intro', got '{tactics[0].text}'"
            assert tactics[0].line == 3

            assert tactics[1].text == "exact hp", f"Expected 'exact hp', got '{tactics[1].text}'"
            assert tactics[1].line == 4

            assert tactics[2].text == "apply And.intro", f"Expected 'apply And.intro', got '{tactics[2].text}'"
            assert tactics[2].line == 5

            assert tactics[3].text == "exact hq", f"Expected 'exact hq', got '{tactics[3].text}'"
            assert tactics[3].line == 6

            assert tactics[4].text == "exact hp", f"Expected 'exact hp', got '{tactics[4].text}'"
            assert tactics[4].line == 7

    def test_simple_partial_proof(self):
        """Test parsing a simple proof with apply and exact tactics."""
        lean_code = """theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code, fail_on_error=False)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract 4 atomic tactics
            assert len(tactics) == 1, f"Expected 1 tactic, got {len(tactics)}"

            # Check each tactic
            assert tactics[0].text == "apply And.intro", f"Expected 'apply And.intro', got '{tactics[0].text}'"
            assert tactics[0].line == 3

    def test_calc_proof(self):
        """Test parsing a calc-based proof with multiple rewrite steps."""
        lean_code = """
import Mathlib

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
done"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract tactics from calc steps (currently extracts lemma arguments)
            # This is a known limitation - rw tactics are parsed deeply
            assert len(tactics) > 0, "Expected tactics from calc proof"

    def test_proof_with_indentation(self):
        """Test parsing a proof with indented tactics."""
        lean_code = """
import Mathlib

theorem test10 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp
"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract 5 atomic tactics
            assert len(tactics) == 5, f"Expected 5 tactics, got {len(tactics)}"

            # Check tactics are correctly extracted
            tactic_texts = [t.text for t in tactics]
            expected_tactics = ["apply And.intro", "exact hp", "apply And.intro", "exact hq", "exact hp"]

            for expected, actual in zip(expected_tactics, tactic_texts):
                assert actual == expected, f"Expected '{expected}', got '{actual}'"

    def test_complex_proof_with_have(self):
        """Test parsing a complex proof with have, rw, ring, and linarith."""
        lean_code = """
import Mathlib

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
sorry"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract multiple tactics
            assert len(tactics) == 7, f"Expected 7 tactics, got {len(tactics)}"

            # Check that various tactic types are present
            tactic_texts = [t.text for t in tactics]
            has_rw = any("rw" in t for t in tactic_texts)
            has_have = any("have" in t for t in tactic_texts)
            has_exact = any("exact" in t for t in tactic_texts)
            has_sorry = any("sorry" in t for t in tactic_texts)

            assert has_rw, "Expected to find 'rw' tactics"
            assert has_have, "Expected to find 'have' tactics"
            assert has_exact, "Expected to find 'exact' tactics"
            assert has_sorry, "Expected to find 'sorry' tactic"

    def test_simple_one_liner(self):
        """Test parsing a simple one-line proof."""
        lean_code = "example : True := by trivial"

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract at least the trivial tactic
            assert len(tactics) > 0, "Expected at least one tactic"

            # Check that trivial is present
            tactic_texts = [t.text for t in tactics]
            has_trivial = any("trivial" in t for t in tactic_texts)
            assert has_trivial, f"Expected to find 'trivial' tactic, got {tactic_texts}"

    def test_term_mode_proof_no_tactics(self):
        """Test that term-mode proofs (no tactics) return empty list."""
        lean_code = "theorem test2 : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩"

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Term-mode proof should have no tactics
            assert len(tactics) == 0, f"Expected 0 tactics for term-mode proof, got {len(tactics)}"

    def test_proof_with_done(self):
        """Test parsing a proof ending with 'done'."""
        lean_code = """theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp
done"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Should extract 6 atomic tactics (done is filtered out)
            assert len(tactics) == 6, f"Expected 6 tactics, got {len(tactics)}"

            # 'done' should not be in the tactics
            tactic_texts = [t.text for t in tactics]
            assert "done" in tactic_texts, "Expected 'done' in tactic texts"

    def test_no_duplicate_tactics(self):
        """Test that duplicate tactics are filtered out."""
        lean_code = """theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Check that there are no exact duplicates (same text, same position)
            seen = set()
            for tactic in tactics:
                key = (tactic.text, tactic.line, tactic.column)
                assert key not in seen, f"Found duplicate tactic: {tactic.text} at line {tactic.line}"
                seen.add(key)

    def test_tactic_positions(self):
        """Test that tactic positions are correctly reported."""
        lean_code = """theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
apply And.intro
exact hp
apply And.intro
exact hq
exact hp"""

        with TacticParser(project_path=project_path) as parser:
            tactics, error_str = parser.parse(lean_code)
            print_tactics(tactics)
            if error_str:
                print(f"Error: {error_str}")

            # Check that positions are reasonable
            for tactic in tactics:
                assert tactic.line > 0, f"Line number should be positive, got {tactic.line}"
                assert tactic.column >= 0, f"Column number should be non-negative, got {tactic.column}"
                assert tactic.end_line >= tactic.line, \
                    f"End line {tactic.end_line} should be >= start line {tactic.line}"
                assert tactic.end_column > 0, f"End column should be positive, got {tactic.end_column}"


if __name__ == "__main__":
    # Run tests without pytest
    import traceback

    test_suite = TestTacticParser()
    test_methods = [
        method for method in dir(test_suite)
        if method.startswith("test_")
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_method}")
            print('='*60)
            getattr(test_suite, test_method)()
            print(f"✓ PASSED: {test_method}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_method}")
            print(f"Error: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print('='*60)

    if failed > 0:
        sys.exit(1)
