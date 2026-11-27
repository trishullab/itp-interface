"""Comprehensive regression tests for Lean 4 declaration parsing, proof
extraction, docstring extraction, `verso_sf_extract`, and `parsing_helpers`
fixes.
"""

import os

from itp_interface.lean.parsing_helpers import LeanDeclType, parse_lean_text
from itp_interface.lean.tactic_parser import (
    RequestType,
    TacticParser,
    get_path_to_tactic_parser_executable,
)
from itp_interface.tools.verso_sf_extract import (
    extract_leading_doc,
    iter_code_blocks,
    split_block_decls,
    split_signature_proof,
    unwrap_solution,
)


def _get_parser() -> TacticParser:
    ver = os.environ.get("LEAN_VERSION", "4.36.0-rc2")
    os.environ["LEAN_VERSION"] = ver
    exe = get_path_to_tactic_parser_executable(ver)
    return TacticParser(parser_path=exe)


class TestLineParserFixes:
    """Regression tests for `LineParser.lean` via `TacticParser(PARSE_THEOREM)`."""

    def test_where_def_and_indented_docstrings_not_truncated(self):
        code = """\
module
public section
namespace Array

/--
`O(|xs| + |ys|)`. Merge arrays `xs` and `ys`.
-/
def merge (lt : α → α → Bool) (xs ys : Array α) : Array α :=
  go (Array.mkEmpty (xs.size + ys.size)) 0 0
where
  /-- Auxiliary definition for `merge`. -/
  go (acc : Array α) (i j : Nat) : Array α :=
    if hi : i ≥ xs.size then acc ++ ys[j:]
    else acc
  termination_by xs.size + ys.size - (i + j)

set_option linter.unusedVariables.funArgs false in
/-- Merge with dedup. -/
def mergeDedup [ord : Ord α] (xs ys : Array α) : Array α :=
  xs ++ ys

end Array
"""
        with _get_parser() as p:
            infos, _ = p.parse(code, fail_on_error=False, parse_type=RequestType.PARSE_THEOREM)
        decls = {i.name: i for i in infos if i.decl_type not in ("namespace", "end", "section", "unknown")}
        assert "merge" in decls
        assert decls["merge"].decl_type == "def"
        assert decls["merge"].namespace == "Array"
        assert "termination_by xs.size + ys.size - (i + j)" in decls["merge"].text
        assert "/-- Auxiliary definition for `merge`. -/" in decls["merge"].text
        assert decls["merge"].doc_string is not None
        assert "Merge arrays" in decls["merge"].doc_string

        assert "mergeDedup" in decls
        assert decls["mergeDedup"].decl_type == "def"
        assert decls["mergeDedup"].namespace == "Array"
        assert decls["mergeDedup"].text.startswith("set_option linter.unusedVariables.funArgs false in")
        assert decls["mergeDedup"].doc_string == "/-- Merge with dedup. -/"

    def test_structure_and_inductive_with_internal_docstrings(self):
        code = """\
namespace Demo

/-- Outer structure doc. -/
@[ext]
structure Box (α : Type) where
  /-- Field doc 1. -/
  val : α
  /-- Field doc 2. -/
  tag : Nat

/-- Outer inductive doc. -/
inductive MySum (α β : Type) : Type
  /-- Left constructor doc. -/
  | protected inl (a : α) : MySum α β
  /-- Right constructor doc. -/
  | protected inr (b : β) : MySum α β

end Demo
"""
        with _get_parser() as p:
            infos, _ = p.parse(code, fail_on_error=False, parse_type=RequestType.PARSE_THEOREM)
        decls = {i.name: i for i in infos if i.decl_type not in ("namespace", "end", "section", "unknown")}
        assert decls["Box"].decl_type == "structure"
        assert decls["Box"].namespace == "Demo"
        assert decls["Box"].doc_string == "/-- Outer structure doc. -/"
        assert "tag : Nat" in decls["Box"].text
        assert decls["MySum"].decl_type == "inductive"
        assert decls["MySum"].doc_string == "/-- Outer inductive doc. -/"
        assert "| protected inr (b : β)" in decls["MySum"].text

    def test_command_in_modifiers_abbrev_opaque_lemma_and_sections(self):
        code = """\
namespace Outer.Inner

section HelperSec
variable (n : Nat)

/-- An abbrev. -/
abbrev MyNat := Nat

/-- An opaque constant. -/
opaque hiddenVal : Nat := 0

end HelperSec

-- Plain line comment (NOT a doc comment!) with UTF-8: α → β ≤ γ — δ
variable (M : Type) in
/-- Doc after variable in. -/
noncomputable def myConn (x : M) : M := x

@[simp] lemma my_simp_lemma (x : Nat) : x = x := rfl

public lemma my_pub_lemma (x : Nat) : x = x := rfl

end Inner
def afterInner : Nat := 1
end Outer
"""
        with _get_parser() as p:
            infos, _ = p.parse(code, fail_on_error=False, parse_type=RequestType.PARSE_THEOREM)
        decls = {i.name: i for i in infos if i.decl_type not in ("namespace", "end", "section", "unknown")}

        assert decls["MyNat"].decl_type == "abbrev"
        assert decls["MyNat"].namespace == "Outer.Inner"
        assert decls["MyNat"].doc_string == "/-- An abbrev. -/"

        assert decls["hiddenVal"].decl_type == "opaque"
        assert decls["hiddenVal"].namespace == "Outer.Inner"

        # After `end HelperSec`, namespace MUST remain `Outer.Inner`!
        assert decls["myConn"].decl_type == "def"
        assert decls["myConn"].namespace == "Outer.Inner"
        assert decls["myConn"].doc_string == "/-- Doc after variable in. -/"
        assert "Plain line comment" not in (decls["myConn"].doc_string or "")
        assert decls["myConn"].text.startswith("variable (M : Type) in")

        assert decls["my_simp_lemma"].decl_type == "lemma"
        assert decls["my_simp_lemma"].namespace == "Outer.Inner"
        assert "lemma my_simp_lemma" in decls["my_simp_lemma"].text

        assert decls["my_pub_lemma"].decl_type == "lemma"
        assert decls["my_pub_lemma"].namespace == "Outer.Inner"
        assert "public lemma my_pub_lemma" in decls["my_pub_lemma"].text

        # After `end Inner`, namespace MUST be `Outer`!
        assert decls["afterInner"].namespace == "Outer"


class TestParsingHelpersFixes:
    def test_primed_identifiers_do_not_break_tokenizer(self):
        txt = """\
/-- Doc for primed theorem. -/
theorem foo' (x' : Nat) : x' = x' := by
  rfl
"""
        res = parse_lean_text(txt)
        assert res.decl_type == LeanDeclType.THEOREM
        assert res.name == "foo" or res.name == "foo'"
        assert res.proof == ":= by\n  rfl"
        assert res.doc_string == "/-- Doc for primed theorem. -/"

    def test_def_body_is_not_split_into_proof(self):
        txt = "def addOne (n : Nat) : Nat := n + 1"
        res = parse_lean_text(txt)
        assert res.decl_type == LeanDeclType.DEF
        assert res.proof is None
        assert ":= n + 1" in (res.text or "")


class TestVersoSfExtractFixes:
    def test_unwrap_multiline_solution_and_workinclass_dedent(self):
        src = """\
def is_weekend (d : Day) : Bool
  := solution!
    (match d with
    | Day.saturday => true -- (paren in comment)
    | _ => false
    )

theorem t1 : True := by
  have h : True := trivial
  workinclass!
    exact h
"""
        unwrapped, had = unwrap_solution(src)
        assert had is True
        assert "solution!" not in unwrapped
        assert "workinclass!" not in unwrapped
        assert "match d with" in unwrapped
        # `exact h` must be dedented to match `have h` (2 spaces)
        assert "\n  exact h" in unwrapped

    def test_persistent_namespace_prose_doc_and_def_not_split(self):
        verso = """\
#doc (Manual) "Test" =>

```lean
namespace MyBool
```

::::exercise (rating := 1) (name := "nand")
Complete the definition of `nand`. It returns {name}`MyBool.true` if either input is false.

```lean
def nand (b1 b2 : Bool) : Bool := solution!(match b1 with | true => !b2 | false => true)

theorem nand_t1 : nand true false = true := solution!(by rfl)
```
::::

```lean -keep
theorem bogus : False := by sorry
```

```lean
end MyBool
```
"""
        blocks = list(iter_code_blocks(verso))
        # `-keep` block must be skipped -> only 3 kept blocks
        assert len(blocks) == 3
        scope_stack = []
        collected = []
        for body, sline, ex, prose, is_imp in blocks:
            decls, scope_stack = split_block_decls(body, sline, scope_stack)
            for dtxt, ds, de, ns in decls:
                doc, rest = extract_leading_doc(dtxt)
                unw, _ = unwrap_solution(rest)
                collected.append((ns, ex, prose, unw))
        assert len(collected) == 2
        assert collected[0][0] == "MyBool"
        assert "Complete the definition of `nand`" in (collected[0][2] or "")
        sig_def, prf_def = split_signature_proof(collected[0][3], "def")
        assert prf_def is None and "match b1 with" in sig_def
        sig_thm, prf_thm = split_signature_proof(collected[1][3], "theorem")
        assert prf_thm == ":= by rfl"
