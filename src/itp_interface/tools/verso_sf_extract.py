#!/usr/bin/env python3
"""Extract Lean declarations from a Verso ``sf-in-lean`` book into a declaration DB.

The Software-Foundations-in-Lean sources are Verso literate-Lean documents: the
Lean lives inside ```` ```lean ... ``` ```` fenced blocks, and student-facing
exercises wrap their reference solution in ``solution!`` markers or
``-- SOLUTION`` / ``-- END SOLUTION`` regions so the distributed version shows
``sorry``:

    ::::exercise (rating := 1) (name := "nand")
    ...prose...
    ```lean
    def nand (b1 : MyBool) (b2 : MyBool) : MyBool
      := solution!(match b1 with | ... )          -- expression form (single or multi-line)
    theorem nand_test1 : ... := solution!(by rfl)
    ```
    ::::

    theorem succ_eq_add_one (n : Nat) : succ n = n + one := by
      solution!                                     -- tactic form
      rewrite [...]; rfl

This module parses the Verso document directly (no Lean build required): it walks
``:::``-directives to attach each ```` ```lean ```` block to its enclosing
``exercise`` (recovering ``name``/``rating``/``optional``/``level``/``manual``)
and extracts the cleaned Verso prose into ``/-- ... -/`` docstrings, tracks
chapter-wide ``namespace``/``section``/``end`` scopes across blocks, splits each
block into individual declarations, **unwraps** all ``solution!`` and
``-- SOLUTION``/``-- END SOLUTION`` forms to recover the reference statement +
proof, splits ``proof`` ONLY for ``theorem``/``lemma``/``example`` (preserving
full bodies for ``def``/``abbrev``/``structure``/``inductive``/``class``), and
writes rows in the exact ``declarations`` schema plus a ``decl_meta`` side table.

````` ```lean +error ``` ````` and ````` ```lean -keep ``` ````` blocks (deliberately
broken or non-kept illustrations) are skipped, as are non-book directives
(``:::instructors``, ``:::dev``, ``:::quiz``, ``:::ignore``, ``:::terse``).

Usage:
    python -m itp_interface.tools.verso_sf_extract \
        --root /path/to/sf-in-lean --volumes LF HL TS \
        --out  /path/to/sf_in_lean.sqlite
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from itp_interface.tools.simple_sqlite import LeanDeclarationDB

DECL_KW = ('theorem', 'lemma', 'def', 'abbrev', 'example', 'instance',
           'inductive', 'coinductive', 'structure', 'class', 'opaque', 'axiom')
PROOF_SPLIT_KW = {'theorem', 'lemma', 'example'}
MODIFIERS = ('noncomputable', 'private', 'protected', 'partial', 'unsafe',
             'scoped', 'local', 'public', 'meta')
IN_PREFIX_KW = ('set_option', 'variable', 'universe', 'open', 'omit', 'include',
                'attribute')
STRUCTURAL = ('namespace', 'section', 'end', 'open', 'variable', 'universe',
              'set_option', 'import', 'attribute', 'macro_rules', 'syntax',
              'notation', 'infixl', 'infixr', 'infix', 'prefix', 'postfix',
              'elab', 'macro', 'declare_syntax_cat', 'register_', 'deriving',
              'export', 'initialize', 'builtin_initialize', 'prelude', 'module',
              'add_decl_doc', 'recommended_spelling')
NOTATION_KW = ('infixl', 'infixr', 'infix', 'prefix', 'postfix', 'notation',
               'syntax', 'macro', 'elab', 'macro_rules', 'declare_syntax_cat')

# Verso directives whose content (both prose and code) is NOT part of the
# compiled book chapters (instructor notes, author dev notes, quizzes, terse
# duplicate summaries, explicitly ignored blocks).
SKIP_CODE_DIRECTIVES = {'instructors', 'dev', 'quiz', 'ignore', 'terse'}
SKIP_PROSE_DIRECTIVES = {
    'instructors', 'dev', 'quiz', 'ignore', 'terse', 'slidebreak',
    'autogradedHole', 'gradeTheorem', 'suppressPreviousHeaderWhenTerse',
    'details', 'hide', 'bnf', 'diagramWithAlt',
}

_RE_DIR_OPEN = re.compile(r'^(:{3,})([A-Za-z]\w*)(.*)$')
_RE_DIR_CLOSE = re.compile(r'^(:{3,})\s*$')
_RE_FENCE = re.compile(r'^(\s*)```([A-Za-z]\w*)(.*)$')
_RE_FENCE_END = re.compile(r'^\s*```\s*$')
_RE_ATTR = re.compile(r'\((\w+)\s*:=\s*("[^"]*"|[^)]*)\)')
_RE_IN_CMD = re.compile(
    r'^(?:' + '|'.join(IN_PREFIX_KW) + r')\b.*\bin\b\s*(?:--.*)?$'
)
_RE_IN_INLINE = re.compile(
    r'^(?:' + '|'.join(IN_PREFIX_KW) + r')\b.*?\bin\s+(?=(?:@\[|/--|'
    + '|'.join(MODIFIERS) + r'\b|' + '|'.join(DECL_KW) + r'\b))'
)


# ---------------------------------------------------------------------------
# Lexical helpers (string, char, comment-aware)
# ---------------------------------------------------------------------------
def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c in "_'"


def _skip_string(s: str, i: int) -> int:
    """Given s[i] == '"', return index just past the closing quote."""
    i += 1
    n = len(s)
    while i < n:
        if s[i] == '\\':
            i += 2
            continue
        if s[i] == '"':
            return i + 1
        i += 1
    return i


def _skip_block_comment(s: str, i: int) -> int:
    """Given s[i:i+2] == '/-', return index just past the matching '-/'."""
    depth = 1
    i += 2
    n = len(s)
    while i < n and depth > 0:
        if s.startswith('/-', i):
            depth += 1
            i += 2
        elif s.startswith('-/', i):
            depth -= 1
            i += 2
        else:
            i += 1
    return i


def _looks_like_char_literal(s: str, i: int) -> int:
    """If s[i] == "'" begins a Lean char literal (not an identifier prime),
    return index just past the closing quote; otherwise return 0."""
    n = len(s)
    if i > 0 and _is_ident_char(s[i - 1]):
        return 0
    if i + 2 < n and s[i + 1] != '\\' and s[i + 1] != '\n' and s[i + 2] == "'":
        return i + 3
    if i + 3 < n and s[i + 1] == '\\' and s[i + 3] == "'":
        return i + 4
    return 0


# ---------------------------------------------------------------------------
# solution! and -- SOLUTION / -- END SOLUTION unwrapping
# ---------------------------------------------------------------------------
_RE_TAC_MARKER_SOLO = re.compile(r'^([ \t]*)(?:solution!|workinclass!|suggested!)[ \t]*(?:--.*)?$')
_RE_TAC_MARKER_INLINE = re.compile(r'^([ \t]*)(?:solution!|workinclass!|suggested!)[ \t]+(.+)$')


def _unwrap_tactic_markers(text: str) -> Tuple[str, bool]:
    """Unwrap ``withPosition(("solution!" | "workinclass!" | "suggested!") tacticSeqIndentGt)``
    blocks, dedenting their indented tactic bodies back to the marker's column
    so Lean 4's whitespace-sensitive tactic blocks remain 100% valid."""
    had = False
    lines = text.split('\n')
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m_solo = _RE_TAC_MARKER_SOLO.match(line)
        if m_solo:
            had = True
            marker_indent = len(m_solo.group(1))
            i += 1
            # Collect all lines belonging to the indented tacticSeqIndentGt block
            block: List[str] = []
            first_body_indent: Optional[int] = None
            while i < n:
                cur = lines[i]
                if cur.strip() == '':
                    block.append(cur)
                    i += 1
                    continue
                cur_indent = len(cur) - len(cur.lstrip(' \t'))
                if cur_indent > marker_indent:
                    if first_body_indent is None:
                        first_body_indent = cur_indent
                    block.append(cur)
                    i += 1
                else:
                    break
            # Trim trailing blank lines from block so we don't dedent outside
            trailing_blanks: List[str] = []
            while block and block[-1].strip() == '':
                trailing_blanks.append(block.pop())
            dedent = (first_body_indent - marker_indent) if first_body_indent is not None else 0
            for bl in block:
                if bl.strip() == '':
                    out.append('')
                elif dedent > 0:
                    cur_indent = len(bl) - len(bl.lstrip(' \t'))
                    remove = min(dedent, cur_indent - marker_indent)
                    out.append(bl[remove:])
                else:
                    out.append(bl)
            out.extend(reversed(trailing_blanks))
            continue

        m_inline = _RE_TAC_MARKER_INLINE.match(line)
        if m_inline:
            had = True
            out.append(m_inline.group(1) + m_inline.group(2))
            i += 1
            continue

        out.append(line)
        i += 1

    res = '\n'.join(out)
    # Catch any remaining inline occurrence (e.g. `:= by solution! rfl`)
    res2 = re.sub(r'(?<![\w.])(?:solution!|workinclass!|suggested!)(?![(\w])[ \t]*', '', res)
    if res2 != res:
        had = True
    return res2, had


def unwrap_solution(text: str) -> Tuple[str, bool]:
    """Unwrap all Software-Foundations solution/in-class markers:

      1. ``solution!(X)`` and multi-line ``solution!\\n  (X)`` -> ``X``
         (paren-balanced, skipping strings, char literals, ``--`` comments,
         and nested ``/- ... -/`` comments).
      2. Tactic-mode ``solution!``, ``workinclass!``, and ``suggested!`` blocks ->
         marker removed and indented body dedented to the marker's column.
      3. ``-- SOLUTION`` and ``-- END SOLUTION`` marker lines -> removed,
         preserving the body verbatim.

    Returns ``(unwrapped_text, had_solution)``.
    """
    had = False

    # 1. Strip `-- SOLUTION` / `-- END SOLUTION` marker lines
    lines = text.split('\n')
    kept_lines: List[str] = []
    for ln in lines:
        st = ln.strip()
        if st in ('-- SOLUTION', '-- END SOLUTION'):
            had = True
            continue
        kept_lines.append(ln)
    text = '\n'.join(kept_lines)

    # 2. Unwrap `solution!(...)` and `solution!\s*(...)`
    out: List[str] = []
    i = 0
    n = len(text)
    tok = 'solution!'
    tok_len = len(tok)

    while i < n:
        c = text[i]
        # Skip comments and strings while searching for `solution!`
        if text.startswith('--', i):
            nl = text.find('\n', i)
            end = n if nl == -1 else nl + 1
            out.append(text[i:end])
            i = end
            continue
        if text.startswith('/-', i):
            end = _skip_block_comment(text, i)
            out.append(text[i:end])
            i = end
            continue
        if c == '"':
            end = _skip_string(text, i)
            out.append(text[i:end])
            i = end
            continue

        if text.startswith(tok, i) and (i == 0 or not _is_ident_char(text[i - 1])):
            after_tok = i + tok_len
            # Check if followed (possibly across whitespace/newline) by `(`
            ws = after_tok
            while ws < n and text[ws] in ' \t\r\n':
                ws += 1
            if ws < n and text[ws] == '(':
                # Parenthesised expression form `solution!(...)` or `solution!\n  (...)`
                had = True
                k = ws + 1
                depth = 1
                start_inner = k
                while k < n and depth > 0:
                    ch = text[k]
                    if text.startswith('--', k):
                        nl = text.find('\n', k)
                        k = n if nl == -1 else nl + 1
                        continue
                    if text.startswith('/-', k):
                        k = _skip_block_comment(text, k)
                        continue
                    if ch == '"':
                        k = _skip_string(text, k)
                        continue
                    if ch == "'":
                        cl = _looks_like_char_literal(text, k)
                        if cl:
                            k = cl
                            continue
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                inner = text[start_inner:k]
                out.append(inner)
                i = k + 1 if k < n else k
                continue

        out.append(c)
        i += 1

    text = ''.join(out)

    # 3. Unwrap tactic-mode `solution!`, `workinclass!`, `suggested!` with dedent
    text, had_tac = _unwrap_tactic_markers(text)
    had = had or had_tac
    return text, had


# ---------------------------------------------------------------------------
# Verso prose cleanup -> Lean /-- ... -/ docstrings
# ---------------------------------------------------------------------------
def clean_verso_prose(lines: List[str]) -> Optional[str]:
    """Convert raw Verso markdown/role prose into clean readable plain text
    suitable for a Lean ``/-- ... -/`` doc-comment."""
    if not lines:
        return None
    text = '\n'.join(lines).strip()
    if not text:
        return None

    # Drop %%% ... %%% metadata blocks
    text = re.sub(r'(?ms)^%%%\s*\n.*?\n%%%\s*$', '', text)
    # Drop Verso `{include ...}` or `{anchor ...}` directives
    text = re.sub(r'\{(?:include|anchor|index|margin)\b[^}]*\}(?:\[[^\]]*\])?', '', text)
    # Convert `{ref "Target"}[label]` or `{link ...}[label]` -> `label`
    text = re.sub(r'\{(?:ref|link|tech|emph|bold)\b[^}]*\}\[([^\]]*)\]', r'\1', text)
    # Convert inline code roles `{name}`foo``, `{lean}`foo``, `{tactic}`foo``, `{kw}`foo``, `{lit}`foo`` -> `` `foo` ``
    text = re.sub(r'\{(?:name|lean|tactic|kw|lit|option|module|attr|syntax)\b[^}]*\}(`+[^`]+`+)', r'\1', text)
    # Convert `{sflVersion}[]` / `{sflYear}[]` -> strip
    text = re.sub(r'\{[A-Za-z_]\w*\}\[\]', '', text)
    # Remove any stray `---` or `/-` / `-/` sequences that could prematurely close `/-- ... -/`
    text = text.replace('/-', '/ -').replace('-/', '- /')
    # Normalize blank lines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    # Skip trivially short or pure-punctuation prose (e.g. "Evaluation:")
    if len(text) < 15:
        return None
    return text


def format_as_doc_comment(prose: str) -> str:
    prose = prose.strip()
    if '\n' in prose or len(prose) > 85:
        return f"/--\n{prose}\n-/"
    return f"/-- {prose} -/"


# ---------------------------------------------------------------------------
# Chapter-wide namespace/section scope stack (exact Lean 4 semantics)
# ---------------------------------------------------------------------------
ScopeFrame = Tuple[bool, List[str]]   # (is_namespace, dotted_components)


def _split_scope_parts(name: str) -> List[str]:
    return [p for p in name.strip().split('.') if p]


def _pop_scope(stack: List[ScopeFrame], end_name: Optional[str]) -> List[ScopeFrame]:
    if not stack:
        return stack
    if not end_name:
        return stack[:-1]
    rem = _split_scope_parts(end_name)
    if not rem:
        return stack[:-1]
    st = [ (is_ns, list(parts)) for is_ns, parts in stack ]
    while rem and st:
        is_ns, parts = st[-1]
        if not parts:
            return stack
        if len(parts) <= len(rem):
            if rem[-len(parts):] == parts:
                st.pop()
                rem = rem[:-len(parts)]
            else:
                return stack
        else:
            if parts[-len(rem):] == rem:
                st[-1] = (is_ns, parts[:-len(rem)])
                rem = []
            else:
                return stack
    return st if not rem else stack


def _active_namespace(stack: List[ScopeFrame]) -> Optional[str]:
    parts: List[str] = []
    for is_ns, ps in stack:
        if is_ns:
            parts.extend(ps)
    return '.'.join(parts) if parts else None


# ---------------------------------------------------------------------------
# Declaration splitting within a code block (preserving chapter scope_stack)
# ---------------------------------------------------------------------------
def _is_in_combinator(stripped: str) -> bool:
    return bool(_RE_IN_CMD.match(stripped) or _RE_IN_INLINE.match(stripped))


def _is_structural_line(stripped: str) -> bool:
    if _is_in_combinator(stripped):
        return False
    w = stripped.split()
    if not w:
        return False
    head0 = re.split(r'[:\s]', w[0])[0]
    if head0 in STRUCTURAL or head0 in NOTATION_KW or w[0].startswith('#'):
        return True
    if w[0] in ('scoped', 'local', 'public', 'private', 'noncomputable') and len(w) > 1:
        second = re.split(r'[\s:]', w[1])[0]
        if second in NOTATION_KW or second in ('section', 'namespace', 'open', 'variable', 'set_option', 'attribute'):
            return True
    return False


def _is_head_starter(stripped: str) -> bool:
    """A col-0 line that can begin a declaration head (keyword / modifier / attr / doc / ... in)."""
    if stripped.startswith('@[') or stripped.startswith('/--'):
        return True
    if _is_in_combinator(stripped):
        return True
    w = stripped.split()
    if not w:
        return False
    if w[0] == 'mutual':
        return True
    return w[0] in DECL_KW or w[0] in MODIFIERS


def _chunk_has_keyword(lines: List[Tuple[str, int]]) -> bool:
    txt = '\n'.join(l for l, _ in lines)
    from_doc = extract_leading_doc(txt)[1]
    kw, _ = detect_kw_name(from_doc)
    return kw != 'unknown'


def _update_comment_depth(line: str, depth: int) -> int:
    """Update nested block comment depth across a single line (string-aware)."""
    i = 0
    n = len(line)
    in_str = False
    while i < n:
        if depth == 0:
            if in_str:
                if line[i] == '\\':
                    i += 2
                    continue
                if line[i] == '"':
                    in_str = False
                i += 1
                continue
            if line.startswith('--', i):
                break
            if line[i] == '"':
                in_str = True
                i += 1
                continue
            if line.startswith('/-', i):
                depth += 1
                i += 2
                continue
            i += 1
        else:
            if line.startswith('/-', i):
                depth += 1
                i += 2
            elif line.startswith('-/', i):
                depth -= 1
                i += 2
            else:
                i += 1
    return depth


def split_block_decls(
    block_lines: List[str],
    block_start_line: int,
    scope_stack: List[ScopeFrame],
) -> Tuple[List[Tuple[str, int, int, Optional[str]]], List[ScopeFrame]]:
    """Split a code block into top-level declarations while updating and returning
    the chapter-wide ``scope_stack``.

    Yields ``(decl_text, start_line, end_line, active_namespace)`` tuples.
    """
    cur: List[Tuple[str, int]] = []
    cur_ns: Optional[str] = None
    comment_depth = 0
    in_mutual = False
    results: List[Tuple[str, int, int, Optional[str]]] = []

    def flush():
        nonlocal cur
        if cur:
            txt = '\n'.join(l for l, _ in cur).rstrip()
            if txt.strip():
                results.append((txt, cur[0][1], cur[-1][1], cur_ns))
        cur = []

    for idx, line in enumerate(block_lines):
        abs_no = block_start_line + idx
        s = line.strip()
        at_col0 = (len(line) - len(line.lstrip())) == 0 and s != ''

        if comment_depth > 0:
            if cur:
                cur.append((line, abs_no))
            comment_depth = _update_comment_depth(line, comment_depth)
            continue

        # Inside a `mutual ... end` block: accumulate until column-0 `end`
        if in_mutual:
            cur.append((line, abs_no))
            comment_depth = _update_comment_depth(line, comment_depth)
            if at_col0 and (s == 'end' or s.startswith('end ')):
                in_mutual = False
                flush()
            continue

        # Skip standalone non-doc block comments (`/- ... -/` or `/-! ... -/`)
        if at_col0 and s.startswith('/-') and not s.startswith('/--'):
            if cur and _chunk_has_keyword(cur):
                flush()
            comment_depth = _update_comment_depth(line, 0)
            continue

        if at_col0 and _is_structural_line(s):
            flush()
            w = s.split()
            # Handle `public section`, `noncomputable section`, `section Foo`, `namespace A.B`, `end [A.B]`
            if w[0] == 'namespace' and len(w) > 1:
                scope_stack.append((True, _split_scope_parts(w[1])))
            elif w[0] == 'section':
                parts = _split_scope_parts(w[1]) if len(w) > 1 and not w[1].startswith('--') else []
                scope_stack.append((False, parts))
            elif len(w) > 1 and w[0] in ('public', 'noncomputable') and w[1] == 'section':
                parts = _split_scope_parts(w[2]) if len(w) > 2 and not w[2].startswith('--') else []
                scope_stack.append((False, parts))
            elif w[0] == 'end':
                end_name = w[1] if len(w) > 1 and not w[1].startswith('--') else None
                scope_stack = _pop_scope(scope_stack, end_name)
            comment_depth = _update_comment_depth(line, 0)
            continue

        if at_col0 and _is_head_starter(s):
            if cur and _chunk_has_keyword(cur):
                flush()
            if not cur:
                cur_ns = _active_namespace(scope_stack)
            cur.append((line, abs_no))
            if s == 'mutual' or s.startswith('mutual '):
                in_mutual = True
            comment_depth = _update_comment_depth(line, 0)
            continue

        # Continuation of the current declaration
        if cur:
            cur.append((line, abs_no))
            comment_depth = _update_comment_depth(line, 0)
        else:
            comment_depth = _update_comment_depth(line, 0)

    flush()
    return results, scope_stack


# ---------------------------------------------------------------------------
# Per-declaration normalization
# ---------------------------------------------------------------------------
def extract_leading_doc(text: str) -> Tuple[Optional[str], str]:
    """Extract a genuine leading `/-- ... -/` doc-comment (including after a
    leading `... in` prefix) while preserving any `... in` prefix in `rest`."""
    s = text.lstrip('\n')
    # Case 1: doc comment at the very start
    if s.lstrip().startswith('/--'):
        start = s.find('/--')
        end = _skip_block_comment(s, start)
        doc = s[start:end].strip()
        rest = (s[:start] + s[end:]).lstrip('\n')
        return doc, rest
    # Case 2: doc comment immediately after leading `... in` line(s)
    lines = s.split('\n')
    i = 0
    while i < len(lines) and _is_in_combinator(lines[i].strip()):
        i += 1
    if i > 0 and i < len(lines) and lines[i].lstrip().startswith('/--'):
        prefix = '\n'.join(lines[:i])
        tail = '\n'.join(lines[i:])
        start = tail.find('/--')
        end = _skip_block_comment(tail, start)
        doc = tail[start:end].strip()
        after = tail[end:].lstrip('\n')
        rest = (prefix.rstrip() + '\n' + after).strip() if after else prefix.strip()
        return doc, rest
    return None, text


def split_signature_proof(text: str, decl_kw: str) -> Tuple[str, Optional[str]]:
    """For ``theorem``/``lemma``/``example``, split at the FIRST top-level proof
    delimiter (``:=``, ``where``, or ``|``) in ascending position order, skipping
    comments, strings, char literals, and nested brackets ``() [] {} ⟨⟩``.
    For all other declaration kinds (``def``, ``abbrev``, ``structure``,
    ``inductive``, ``class``, ``instance``, ``opaque``, ``axiom``, ``mutual``),
    keep the complete body in ``text`` and return ``proof = None``."""
    if decl_kw not in PROOF_SPLIT_KW:
        return text.rstrip(), None

    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith('--', i):
            nl = text.find('\n', i)
            i = n if nl == -1 else nl + 1
            continue
        if text.startswith('/-', i):
            i = _skip_block_comment(text, i)
            continue
        if c == '"':
            i = _skip_string(text, i)
            continue
        if c == "'":
            cl = _looks_like_char_literal(text, i)
            if cl:
                i = cl
                continue
        if c in '([{⟨':
            depth += 1
            i += 1
            continue
        if c in ')]}⟩':
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            if text.startswith(':=', i):
                prev_ok = i == 0 or text[i - 1] != ':'
                next_ok = i + 2 >= n or text[i + 2] != '='
                if prev_ok and next_ok:
                    return text[:i].rstrip(), text[i:].strip()
                i += 2
                continue
            if text.startswith('where', i):
                prev_ok = i == 0 or not _is_ident_char(text[i - 1])
                next_ok = i + 5 >= n or not _is_ident_char(text[i + 5])
                if prev_ok and next_ok:
                    return text[:i].rstrip(), text[i:].strip()
            if c == '|':
                prev_ok = i == 0 or text[i - 1] not in '|<[{'
                next_ok = i + 1 >= n or text[i + 1] not in '|>-]}'
                if prev_ok and next_ok:
                    return text[:i].rstrip(), text[i:].strip()
        i += 1
    return text.rstrip(), None


def detect_kw_name(text: str) -> Tuple[str, Optional[str]]:
    # Skip leading `... in` lines
    lines = text.lstrip().split('\n')
    idx_ln = 0
    while idx_ln < len(lines) and _is_in_combinator(lines[idx_ln].strip()):
        m_inline = _RE_IN_INLINE.match(lines[idx_ln].strip())
        if m_inline:
            lines[idx_ln] = lines[idx_ln].strip()[m_inline.end():]
            break
        idx_ln += 1
    s = '\n'.join(lines[idx_ln:]).lstrip()

    # Skip leading doc-comment if still present
    if s.startswith('/--'):
        end = _skip_block_comment(s, 0)
        s = s[end:].lstrip()

    # Strip leading `@[...]` attributes
    while s.startswith('@['):
        d = 0
        in_str = False
        for idx, ch in enumerate(s):
            if ch == '"' and (idx == 0 or s[idx - 1] != '\\'):
                in_str = not in_str
            elif not in_str:
                if ch == '[':
                    d += 1
                elif ch == ']':
                    d -= 1
                    if d == 0:
                        s = s[idx + 1:].lstrip()
                        break
        else:
            break

    toks = s.split()
    k = 0
    while k < len(toks) and toks[k] in MODIFIERS:
        k += 1
    if k >= len(toks):
        return 'unknown', None
    kw = toks[k]
    if kw == 'mutual':
        return 'mutual', None
    if kw not in DECL_KW:
        return 'unknown', None
    if kw == 'example':
        return 'example', None
    name = toks[k + 1] if k + 1 < len(toks) else None
    if name:
        # Anonymous instance (e.g. `instance : Foo Bar`) or binder directly after kw
        if name[0] in '(:{[⟨':
            return kw, None
        if name.startswith('«'):
            end_q = s.find('»', s.find('«'))
            if end_q != -1:
                return kw, s[s.find('«'):end_q + 1]
        name = re.split(r'[\s({:\[⟨]', name)[0]
        name = name.rstrip(':')
        if not name:
            name = None
    return kw, name


# ---------------------------------------------------------------------------
# Verso document walk (code blocks + prose + importBlocks)
# ---------------------------------------------------------------------------
def parse_attrs(rest: str) -> Dict[str, str]:
    out = {}
    for m in _RE_ATTR.finditer(rest):
        key, val = m.group(1), m.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        out[key] = val
    return out


def iter_code_blocks(source: str):
    """Walk a Verso document and yield
    ``(block_lines, start_line, exercise_attrs_or_None, prose_doc_or_None, is_import_block)``
    for each kept ```` ```lean ```` (or ```` ```importBlock ````) block.

    Skips ``+error`` and ``-keep`` blocks, ``leanTerm``/``leanOutput`` blocks,
    and blocks nested inside ``:::instructors``, ``:::dev``, ``:::quiz``,
    ``:::ignore``, or ``:::terse``.
    """
    lines = source.split('\n')
    # Stack entries: (colons, directive_name, attrs, exercise_prose_lines)
    stack: List[Tuple[int, str, Dict[str, str], List[str]]] = []
    # Prose lines immediately preceding the next non-exercise code block
    pending_prose: List[str] = []
    # Skip header before `#doc`
    in_doc = '#doc' not in source
    in_meta = False
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not in_doc:
            if stripped.startswith('#doc'):
                in_doc = True
            i += 1
            continue

        if stripped == '%%%':
            in_meta = not in_meta
            i += 1
            continue
        if in_meta:
            i += 1
            continue

        # Code fence open
        mf = _RE_FENCE.match(line)
        if mf:
            lang = mf.group(2).strip()
            info = mf.group(3).strip()
            info_toks = info.split()
            start = i + 1
            body: List[str] = []
            i += 1
            while i < n and not _RE_FENCE_END.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # past closing ```

            in_skip_dir = any(name in SKIP_CODE_DIRECTIVES for _, name, _, _ in stack)
            if in_skip_dir:
                continue

            if lang == 'importBlock':
                yield body, start + 1, None, None, True
                pending_prose = []
                continue

            if lang != 'lean':
                continue
            if '+error' in info_toks or '-keep' in info_toks:
                continue

            # Find enclosing exercise (if any)
            ex_attrs = None
            ex_prose_lines: List[str] = []
            for _, name, attrs, eprose in reversed(stack):
                if name == 'exercise':
                    ex_attrs = attrs
                    ex_prose_lines = eprose
                    break

            if ex_attrs is not None:
                prose_text = clean_verso_prose(ex_prose_lines)
            else:
                prose_text = clean_verso_prose(pending_prose)

            # Reset pending non-exercise prose once consumed by a code block
            pending_prose = []
            yield body, start + 1, ex_attrs, prose_text, False
            continue

        # Directive close `:::`
        mc = _RE_DIR_CLOSE.match(line)
        if mc and stack:
            colons = len(mc.group(1))
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == colons:
                    del stack[j:]
                    break
            i += 1
            continue

        # Directive open `:::name ...`
        mo = _RE_DIR_OPEN.match(line)
        if mo:
            colons, name, rest = len(mo.group(1)), mo.group(2), mo.group(3)
            if name == 'exercise':
                pending_prose = []
            stack.append((colons, name, parse_attrs(rest), []))
            i += 1
            continue

        # Section heading resets pending non-exercise prose
        if stripped.startswith('#'):
            pending_prose = []
            i += 1
            continue

        # Accumulate prose if not inside a skipped prose directive
        if not any(name in SKIP_PROSE_DIRECTIVES for _, name, _, _ in stack):
            # If inside an exercise, append to that exercise's prose buffer
            in_ex = False
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][1] == 'exercise':
                    stack[j][3].append(line)
                    in_ex = True
                    break
            if not in_ex:
                pending_prose.append(line)

        i += 1


# ---------------------------------------------------------------------------
# Official chapter discovery & plain-Lean chapter emission
# ---------------------------------------------------------------------------
def _discover_volume_files(root: str, volume: str) -> List[str]:
    """Return ordered chapter ``.lean`` filenames for ``volume`` using ``<VOL>.lean``
    when available (skipping ``Scratch.lean`` and non-chapter files)."""
    vol_root = os.path.join(root, f"{volume}.lean")
    vdir = os.path.join(root, volume)
    if not os.path.isdir(vdir):
        return []
    if os.path.isfile(vol_root):
        ordered: List[str] = []
        with open(vol_root, encoding='utf-8') as fh:
            for line in fh:
                m = re.match(r'^\s*import\s+' + re.escape(volume) + r'\.(\w+)', line)
                if m:
                    fn = f"{m.group(1)}.lean"
                    if os.path.isfile(os.path.join(vdir, fn)) and fn not in ordered:
                        ordered.append(fn)
        if ordered:
            return ordered
    return sorted(
        fn for fn in os.listdir(vdir)
        if fn.endswith('.lean') and fn != 'Scratch.lean'
    )


def _extract_verso_header_imports(source: str, volume: str) -> List[str]:
    """Extract Lean module imports from a Verso chapter's header (excluding SFLMeta)."""
    imports: List[str] = []
    for line in source.split('\n'):
        st = line.strip()
        if st.startswith('#doc'):
            break
        m = re.match(r'^import\s+([A-Za-z0-9_.]+)', st)
        if m:
            mod = m.group(1)
            if mod == 'SFLMeta' or mod.startswith('SFLMeta.') or mod == 'VersoManual':
                continue
            imports.append(mod)
    return imports


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def extract_file(
    db: LeanDeclarationDB,
    meta_rows: list,
    root: str,
    rel_path: str,
    volume: str,
    chapter: str,
    stats: dict,
    emit_lean_dir: Optional[str] = None,
):
    with open(os.path.join(root, rel_path), encoding='utf-8') as fh:
        source = fh.read()
    module_name = f"{volume}.{chapter}"
    db.get_or_create_file(rel_path, module_name)

    is_verso = '#doc' in source
    scope_stack: List[ScopeFrame] = []
    seen = set()
    emitted_chunks: List[str] = []

    if is_verso:
        header_imps = _extract_verso_header_imports(source, volume)
        for imp in header_imps:
            emitted_chunks.append(f"import {imp}")
        blocks = list(iter_code_blocks(source))
    else:
        # Plain Lean helper file (not Verso)
        blocks = [(source.split('\n'), 1, None, None, False)]

    for body, start_line, ex, prose_text, is_imp_block in blocks:
        raw_block = '\n'.join(body)
        if is_imp_block:
            emitted_chunks.append(raw_block)
            continue

        unwrapped_block, _ = unwrap_solution(raw_block)
        emitted_chunks.append(unwrapped_block)

        decls_in_block, scope_stack = split_block_decls(body, start_line, scope_stack)
        for idx_in_block, (decl_text, dstart, dend, namespace) in enumerate(decls_in_block):
            doc, rest = extract_leading_doc(decl_text)
            unwrapped, had_sol = unwrap_solution(rest)
            kw, name = detect_kw_name(unwrapped)
            if kw == 'unknown':
                stats['skipped_unknown'] += 1
                continue

            # Attach Verso prose as a `/-- ... -/` doc_string when no inline doc exists:
            #  - inside an exercise: attach the exercise spec to every declaration in
            #    the exercise (or its primary declarations)
            #  - outside an exercise: attach the immediately preceding prose to the
            #    first declaration of the block
            doc_source = 'inline' if doc else None
            if doc is None and prose_text:
                if ex is not None or idx_in_block == 0:
                    doc = format_as_doc_comment(prose_text)
                    doc_source = 'exercise_prose' if ex is not None else 'chapter_prose'

            sig, proof = split_signature_proof(unwrapped, kw)
            is_sorry = 1 if (
                (proof and re.search(r'(^|\b)sorry\b', proof))
                or (proof is None and re.search(r'(^|\b)sorry\b', sig))
            ) else 0

            if name is None:
                name = f"_{kw}@{dstart}"
            key = (name, namespace, rel_path, module_name)
            if key in seen:
                name = f"{name}@{dstart}"
                key = (name, namespace, rel_path, module_name)
            seen.add(key)

            decl_id = db.get_or_create_decl_id(name, namespace, rel_path, module_name)
            db.upsert_declaration_full_info(
                decl_id=decl_id, name=name, namespace=namespace,
                file_path=rel_path, module_name=module_name,
                decl_type=kw, text=sig, line=dstart, column=0,
                end_line=dend, end_column=0, doc_string=doc, proof=proof,
            )
            stats['decls'] += 1
            if doc:
                stats['with_doc'] += 1

            # metadata
            meta_rows.append((decl_id, 'volume', volume))
            meta_rows.append((decl_id, 'chapter', chapter))
            meta_rows.append((decl_id, 'in_exercise', '1' if ex is not None else '0'))
            meta_rows.append((decl_id, 'has_solution', '1' if had_sol else '0'))
            meta_rows.append((decl_id, 'is_sorry', str(is_sorry)))
            if doc_source:
                meta_rows.append((decl_id, 'doc_source', doc_source))
            if ex is not None:
                stats['exercise_decls'] += 1
                for k in ('name', 'rating', 'optional', 'level', 'manual', 'checkVisibility'):
                    if k in ex:
                        meta_rows.append((decl_id, 'exercise_' + k, ex[k]))

    if emit_lean_dir is not None and emitted_chunks:
        out_lean = os.path.join(emit_lean_dir, rel_path)
        os.makedirs(os.path.dirname(out_lean), exist_ok=True)
        with open(out_lean, 'w', encoding='utf-8') as out_fh:
            out_fh.write('\n\n'.join(c.rstrip() for c in emitted_chunks if c.strip()) + '\n')


def extract(
    root: str,
    volumes: List[str],
    out_db: str,
    log=print,
    emit_lean_dir: Optional[str] = None,
) -> dict:
    if os.path.exists(out_db):
        os.remove(out_db)
    os.makedirs(os.path.dirname(os.path.abspath(out_db)) or '.', exist_ok=True)
    db = LeanDeclarationDB(out_db)
    db.conn.executescript(
        "CREATE TABLE IF NOT EXISTS decl_meta (decl_id TEXT, key TEXT, value TEXT);"
        "CREATE INDEX IF NOT EXISTS idx_decl_meta_declid ON decl_meta(decl_id);"
        "CREATE INDEX IF NOT EXISTS idx_decl_meta_key ON decl_meta(key);"
    )
    stats = {'files': 0, 'decls': 0, 'with_doc': 0, 'exercise_decls': 0, 'skipped_unknown': 0}
    meta_rows: list = []
    for vol in volumes:
        for fn in _discover_volume_files(root, vol):
            rel = os.path.join(vol, fn)
            chapter = fn[:-5]
            extract_file(
                db, meta_rows, root, rel, vol, chapter, stats,
                emit_lean_dir=emit_lean_dir,
            )
            stats['files'] += 1
    db.conn.executemany("INSERT INTO decl_meta (decl_id, key, value) VALUES (?,?,?)", meta_rows)
    db.conn.commit()
    nmeta = db.conn.execute("SELECT COUNT(*) FROM decl_meta").fetchone()[0]
    db.conn.close()
    log(
        f"[verso_sf_extract] files={stats['files']} declarations={stats['decls']} "
        f"with_doc={stats['with_doc']} exercise_decls={stats['exercise_decls']} "
        f"decl_meta={nmeta} skipped_unknown={stats['skipped_unknown']}"
    )
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--root', required=True, help='sf-in-lean checkout root')
    ap.add_argument('--volumes', nargs='+', default=['LF', 'HL', 'TS'])
    ap.add_argument('--out', required=True, help='output sqlite path')
    ap.add_argument(
        '--emit-lean-dir', default=None,
        help='optional directory to emit standalone unwrapped .lean chapter files',
    )
    args = ap.parse_args(argv)
    extract(args.root, args.volumes, args.out, emit_lean_dir=args.emit_lean_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
