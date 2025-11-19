import sys
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Set

class LeanDeclType(Enum):
    LEMMA = "lemma"
    THEOREM = "theorem"
    DEF = "def"
    DEFINITION = "definition"
    STRUCTURE = "structure"
    CLASS = "class"
    INDUCTIVE = "inductive"
    INSTANCE = "instance"
    ABBREV = "abbrev"
    ABBREVIATION = "abbreviation"
    AXIOM = "axiom"
    EXAMPLE = "example"
    OPAQUE = "opaque"
    CONSTANT = "constant"
    MUTUAL = "mutual"
    UNKNOWN = "unknown"

@dataclass
class LeanParseResult:
    decl_type: LeanDeclType
    text_before: Optional[str] = None
    doc_string: Optional[str] = None
    text: Optional[str] = None
    proof: Optional[str] = None

class LeanDeclParser:
    """
    Parses Lean 4 declarations to separate context, docstrings, 
    declaration headers, and proofs/bodies.
    """
    
    # Keywords that mark the start of a declaration
    # We generate this set from the Enum values for fast lookup
    DECL_KEYWORDS = {m.value for m in LeanDeclType if m != LeanDeclType.UNKNOWN}

    # Types for which we should NOT attempt to extract a proof/body
    NO_PROOF_TYPES = {
        LeanDeclType.INDUCTIVE, 
        LeanDeclType.MUTUAL, 
        LeanDeclType.STRUCTURE,
        LeanDeclType.CLASS
    }

    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.tokens = [] 
        self.docstring_range = None # (start, end)
        self.docstring_content = None
        
        # Key Indices and Info
        self.decl_start = -1
        self.proof_start = -1
        self.context_end = 0
        self.decl_type: LeanDeclType = LeanDeclType.UNKNOWN

    def parse(self) -> LeanParseResult:
        self._tokenize()
        self._analyze_structure()
        return self._construct_result()

    def _tokenize(self):
        """
        Scans text to find tokens, respecting comments, strings, and nesting.
        """
        i = 0
        # States
        NORMAL = 0
        IN_STRING = 1
        IN_CHAR = 2
        
        state = NORMAL
        nesting = 0 # () [] {}
        
        while i < self.n:
            # 1. Handle Comments (Line and Block)
            if state == NORMAL:
                if self.text.startswith("--", i):
                    # Line comment
                    end_line = self.text.find('\n', i)
                    if end_line == -1: end_line = self.n
                    i = end_line
                    continue
                
                if self.text.startswith("/-", i):
                    # Block comment
                    is_doc = self.text.startswith("/--", i)
                    start_idx = i
                    
                    # Find end of block comment (handle nesting)
                    depth = 1
                    i += 2
                    while i < self.n and depth > 0:
                        if self.text.startswith("/-", i):
                            depth += 1
                            i += 2
                        elif self.text.startswith("-/", i):
                            depth -= 1
                            i += 2
                        else:
                            i += 1
                    
                    if is_doc and self.docstring_range is None:
                        self.docstring_range = (start_idx, i)
                        self.docstring_content = self.text[start_idx:i]
                    continue

            # 2. Handle Strings/Chars
            if state == NORMAL:
                if self.text[i] == '"':
                    state = IN_STRING
                    i += 1
                    continue
                if self.text[i] == "'":
                    state = IN_CHAR
                    i += 1
                    continue
            
            elif state == IN_STRING:
                if self.text[i] == '\\': i += 2; continue
                if self.text[i] == '"': state = NORMAL; i += 1; continue
                i += 1
                continue
                
            elif state == IN_CHAR:
                if self.text[i] == '\\': i += 2; continue
                if self.text[i] == "'": state = NORMAL; i += 1; continue
                i += 1
                continue

            # 3. Handle Structure Tokens in NORMAL state
            char = self.text[i]
            
            # Nesting tracking
            if char in "([{": nesting += 1
            elif char in ")]}": nesting = max(0, nesting - 1)
            
            # Token detection
            if nesting == 0:
                # Check for 'in' keyword (standalone)
                if self._is_keyword_at(i, "in"):
                    self.tokens.append(("IN", i, i+2))
                    i += 2
                    continue
                
                # Check for Declaration Keywords
                match_kw = self._match_any_keyword(i, self.DECL_KEYWORDS)
                if match_kw:
                    kw, length = match_kw
                    self.tokens.append(("KW", i, i+length))
                    i += length
                    continue
                
                # Check for Attribute Start
                if self.text.startswith("@[", i):
                    self.tokens.append(("ATTR", i, i+2))
                    i += 2
                    nesting += 1 # The '[' counts as nesting
                    continue
                
                # Check for Proof Starters
                if self.text.startswith(":=", i):
                    self.tokens.append(("PROOF", i, i+2))
                    i += 2
                    continue
                if self.text.startswith("where", i) and self._is_word_boundary(i+5):
                    self.tokens.append(("PROOF", i, i+5))
                    i += 5
                    continue
                if char == '|':
                    self.tokens.append(("PROOF", i, i+1))
                    i += 1
                    continue
            
            i += 1

    def _analyze_structure(self):
        """
        Interpret the token stream to find split points.
        """
        candidate_decl = -1
        decl_keyword_str = None
        
        # Pass 1: Find Declaration Start
        for t_type, t_start, t_end in self.tokens:
            if t_type == "IN":
                candidate_decl = -1 # Reset candidate
                decl_keyword_str = None
            
            elif t_type == "KW":
                if candidate_decl == -1:
                    candidate_decl = t_start
                if decl_keyword_str is None:
                    decl_keyword_str = self.text[t_start:t_end]
            
            elif t_type == "ATTR":
                if candidate_decl == -1:
                    candidate_decl = t_start
                    
        if candidate_decl != -1:
            self.decl_start = candidate_decl
            
            # Resolve Enum Type
            if decl_keyword_str:
                try:
                    self.decl_type = LeanDeclType(decl_keyword_str)
                except ValueError:
                    self.decl_type = LeanDeclType.UNKNOWN
            else:
                self.decl_type = LeanDeclType.UNKNOWN
            
            # Pass 2: Find Proof Start
            skip_proof = self.decl_type in self.NO_PROOF_TYPES
            
            if not skip_proof:
                for t_type, t_start, t_end in self.tokens:
                    if t_start > self.decl_start and t_type == "PROOF":
                        self.proof_start = t_start
                        break
        else:
            # Fallback: No declaration found
            pass

    def _construct_result(self) -> LeanParseResult:
        # Case 1: No declaration found
        if self.decl_start == -1:
            return LeanParseResult(
                decl_type=LeanDeclType.UNKNOWN,
                text_before=self.text
            )

        # Case 2: Declaration found
        split_idx = self.decl_start
        
        decl_end = self.n
        proof_content = None
        if self.proof_start != -1:
            decl_end = self.proof_start
            proof_content = self.text[self.proof_start:].strip()
        
        raw_pre = self.text[:split_idx]
        raw_decl = self.text[split_idx:decl_end]
        doc_content = None

        # Handle Docstring Extraction logic
        if self.docstring_range:
            ds_start, ds_end = self.docstring_range
            doc_content = self.text[ds_start:ds_end]
            
            if ds_start < split_idx:
                # Remove docstring from raw_pre
                pre_part1 = self.text[:ds_start]
                pre_part2 = self.text[ds_end:split_idx]
                raw_pre = pre_part1 + pre_part2

        # Final cleanup
        text_before = raw_pre.strip() or None
        text = raw_decl.strip() or None
        doc = doc_content or None
        proof = proof_content or None

        return LeanParseResult(
            decl_type=self.decl_type,
            text_before=text_before,
            doc_string=doc,
            text=text,
            proof=proof
        )

    # --- Helpers ---
    def _is_keyword_at(self, idx, kw):
        if not self.text.startswith(kw, idx): return False
        return self._is_word_boundary(idx + len(kw))

    def _match_any_keyword(self, idx, keywords):
        if not self.text[idx].isalpha(): return None
        j = idx
        while j < self.n and (self.text[j].isalnum() or self.text[j] == '_'):
            j += 1
        word = self.text[idx:j]
        if word in keywords:
            return word, len(word)
        return None

    def _is_word_boundary(self, idx):
        if idx >= self.n: return True
        c = self.text[idx]
        return not (c.isalnum() or c == '_')


def parse_lean_text(text: str) -> LeanParseResult:
    parser = LeanDeclParser(text)
    return parser.parse()