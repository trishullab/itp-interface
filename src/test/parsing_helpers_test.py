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
            )
        ]

        print(f"\n{'TYPE':<15} | {'TEXT BEFORE':<20} | {'DOC':<10} | {'TEXT':<20} | {'PROOF':<15}")
        print("-" * 105)

        for name, inp in test_cases:
            res = parse_lean_text(inp)

            # Access fields via dot notation
            tp = res.decl_type.value
            tb = (res.text_before or "").replace('\n', '\\n')
            ds = (res.doc_string or "")
            tx = (res.text or "").replace('\n', '\\n')
            pf = (res.proof or "").replace('\n', '\\n')

            if len(tb) > 18: tb = tb[:18] + "..."
            if len(ds) > 8: ds = "/--...-/"
            if len(tx) > 18: tx = tx[:18] + "..."
            if len(pf) > 12: pf = pf[:12] + "..."

            print(f"{tp:<15} | {tb:<20} | {ds:<10} | {tx:<20} | {pf:<15}")

            # Basic assertions to verify parsing works
            self.assertIsNotNone(res.decl_type)
            self.assertIsInstance(res.decl_type, LeanDeclType)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
