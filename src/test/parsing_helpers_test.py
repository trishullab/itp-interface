import unittest
from itp_interface.lean.parsing_helpers import parse_lean_text, LeanDeclType


class ParsingHelpersTest(unittest.TestCase):
    def test_lean_declaration_parsing(self):
        """Test parsing of various Lean 4 declaration types"""
        test_cases = [
            (
                "Simple Theorem",
                "@[simp] lemma foo (x : Nat) : x = x := rfl"
            ),
            (
                "Context and Doc",
                "open Algebra.TensorProduct in\n/-- Doc -/\ntheorem left_of_tensor [Module R] : True where\n  out := sorry"
            ),
            (
                "Inductive (No Proof)",
                "/-- The base e -/\ninductive ExBase : Type\n| A\n| B"
            ),
            (
                "Structure (No Proof)",
                "structure MyStruct where\n  field1 : Nat\n  field2 : Int"
            ),
            (
                "Mutual (No Proof)",
                "mutual\n  inductive A | a\n  inductive B | b\nend"
            ),
            (
                "Run Cmd (Fallback)",
                "open Lean in\nrun_cmd Command.liftTermElabM do\n  logInfo \"hi\""
            ),
            (
                "Inductive Nat (hard case)", 
                "set_option genCtorIdx false in\n/--\nThe natural numbers, starting at zero.\n\nThis type is special-cased by both the kernel and the compiler, and overridden with an efficient\nimplementation. Both use a fast arbitrary-precision arithmetic library (usually\n[GMP](https://gmplib.org/)); at runtime, `Nat` values that are sufficiently small are unboxed.\n-/\ninductive Nat where\n  /--\n  Zero, the smallest natural number.\n\n  Using `Nat.zero` explicitly should usually be avoided in favor of the literal `0`, which is the\n  [simp normal form](lean-manual://section/simp-normal-forms).\n  -/\n  | zero : Nat\n  /--\n  The successor of a natural number `n`.\n\n  Using `Nat.succ n` should usually be avoided in favor of `n + 1`, which is the [simp normal\n  form](lean-manual://section/simp-normal-forms).\n  -/\n  | succ (n : Nat) : Nat\n"
            )
        ]

        print(f"{'TYPE':<12} | {'NAME':<10} | {'TEXT BEFORE':<20} | {'DOC':<10} | {'TEXT':<20} | {'PROOF':<15}")
        print("-" * 115)

        for test_name, inp in test_cases:
            res = parse_lean_text(inp)
            
            tp = res.decl_type.value
            nm = res.name or "<None>"
            tb = (res.text_before or "").replace('\n', '\\n')
            ds = (res.doc_string or "")
            tx = (res.text or "").replace('\n', '\\n')
            pf = (res.proof or "").replace('\n', '\\n')
            
            if len(tb) > 18: tb = tb[:18] + "..."
            if len(ds) > 8: ds = "/--...-/"
            if len(tx) > 18: tx = tx[:18] + "..."
            if len(pf) > 12: pf = pf[:12] + "..."

            print(f"{tp:<12} | {nm:<10} | {tb:<20} | {ds:<10} | {tx:<20} | {pf:<15}")

            # Basic assertions to verify parsing works
            self.assertIsNotNone(res.decl_type)
            self.assertIsInstance(res.decl_type, LeanDeclType)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
