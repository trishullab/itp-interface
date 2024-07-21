#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import re
import typing
from itp_interface.lean_server.lean_context import Obligation, ProofContext

class Lean3Utils:
    lean_internal_lib_cmd = "elan which lean"
    theorem_lemma_search_regex = re.compile(r"(theorem|lemma) ([\w+|\d+]*) ([\S|\s]*?):=")
    proof_context_separator = "⊢"
    proof_context_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
    goal_regex = rf"([\s|\S]*?){proof_context_separator}([\s|\S]*)"

    def remove_comments(text: str) -> str:
        # NOTE: This will ONLY work correctly if the comments are well-formed
        # we will need stack based parsing to handle incomplete comments, this faster
        # when we know that the file compiles
        # Remove comments
        #1. First remove all nested comments
        #2. Then remove all single line comments
        # Comments are of the form:
        # 1. /- ... -/
        # 2. -- ...
        # Let's do 1
        # First, let's find all the comments
        new_text = []
        idx = 0
        while idx < len(text):
            if idx < len(text) - 1 and text[idx] == '/' and text[idx+1] == '-':
                # We found a comment
                # Find the end of the comment
                end_of_comment_idx = idx + 2
                while end_of_comment_idx < len(text) and \
                    not (text[end_of_comment_idx] == '-' and \
                    end_of_comment_idx + 1 < len(text) and \
                    text[end_of_comment_idx + 1] == '/'):
                    end_of_comment_idx += 1
                if end_of_comment_idx >= len(text):
                    # Unfinished comment
                    new_text.extend(text[idx:end_of_comment_idx])
                    idx = end_of_comment_idx
                else:
                    # Remove the comment
                    idx = end_of_comment_idx + 2
            if idx < len(text):
                new_text.append(text[idx])
                idx += 1
        text = "".join(new_text)
        new_text = []
        # Now let's do 2
        idx = 0
        while idx < len(text):
            if idx < len(text) - 1 and text[idx] == '-' and text[idx+1] == '-':
                # We found a comment
                # Find the end of the comment
                end_of_comment_idx = idx + 2
                while end_of_comment_idx < len(text) and text[end_of_comment_idx] != '\n':
                    end_of_comment_idx += 1
                if end_of_comment_idx >= len(text):
                    # Unfinished comment
                    new_text.extend(text[idx:end_of_comment_idx])
                # Remove the comment
                idx = end_of_comment_idx
            if idx < len(text):
                new_text.append(text[idx])
                idx += 1
        text = "".join(new_text)
        return text
    
    def get_lean_root_path() -> str:
        lean_exe = os.popen(Lean3Utils.lean_internal_lib_cmd).read().strip()
        lean_bin_path = os.path.dirname(lean_exe)
        lean_root_path = os.path.dirname(lean_bin_path)
        return lean_root_path

    def get_lean_lib_path() -> str:
        lean_root_path = Lean3Utils.get_lean_root_path()
        lean_lib_path = os.path.join(lean_root_path, "lib", "lean", "library")
        return lean_lib_path
    
    def find_theorems_with_namespaces(text: str) -> typing.List[typing.Tuple[str, str, str]]:
        idx = 0
        theorems = []
        lines = text.split('\n')
        current_namespace = None
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("namespace"):
                current_namespace = line[len("namespace"):].strip()
                # Find the end of the namespace
                end_of_namespace_idx = idx + 1
                end_line = lines[end_of_namespace_idx]
                while end_line is not None and not end_line.startswith("end") and not end_line.endswith(current_namespace):
                    end_of_namespace_idx += 1
                    end_line = lines[end_of_namespace_idx] if end_of_namespace_idx < len(lines) else None
                if end_line is not None:
                    namespace_content = " ".join(lines[idx:end_of_namespace_idx+1])
                    for name, dfn in Lean3Utils.find_theorems(namespace_content):
                        theorems.append((current_namespace, name, dfn))
                idx = end_of_namespace_idx + 1
            else:
                idx += 1
        return theorems
    
    def find_theorems(text: str) -> typing.List[typing.Tuple[str, str]]:
        matches = Lean3Utils.theorem_lemma_search_regex.findall(text)
        theorems = []
        for match in matches:
            name = str(match[1]).strip()
            dfn = str(match[2]).strip()
            name = name.strip(':')
            dfn = dfn.strip(':')
            theorems.append((name, dfn))
        return theorems

    def parse_proof_context_human_readable(proof_context_str: str) -> ProofContext:
        if len(proof_context_str) == 0 and Lean3Utils.proof_context_separator not in proof_context_str:
            return None
        if proof_context_str == "no goals":
            return ProofContext.empty()
        proof_context_str = proof_context_str.strip()
        proof_context_str += "\n\n"
        all_matches = re.findall(Lean3Utils.proof_context_regex, proof_context_str, re.MULTILINE)
        goal_strs = []
        total_goal_cnt = 0
        for _, goal_cnt, goal_str in all_matches:
            if len(goal_cnt) > 0:
               total_goal_cnt = int(goal_cnt)
            goal_str = goal_str.strip()
            goal_strs.append(goal_str)
        if total_goal_cnt > 0:
            assert len(goal_strs) == total_goal_cnt, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
        else:
            assert len(goal_strs) == 1, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
            total_goal_cnt = 1
        assert len(goal_strs) == total_goal_cnt, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
        goals = []
        for goal_str in goal_strs:
            goal = Lean3Utils.parse_goal(goal_str)
            goals.append(goal)
        return ProofContext(goals, [], [], [])

    def parse_goal(goal_str: str):
        goal_str = goal_str.strip()
        goal = ""
        hyps_goals = re.findall(Lean3Utils.goal_regex, goal_str, re.MULTILINE)
        assert len(hyps_goals) == 1, f"Found more than one goal in the goal string: {goal_str}"
        hypotheses_str, goal = hyps_goals[0]
        hypotheses_str = hypotheses_str.strip()
        goal = goal.strip()
        hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
        goal = Obligation(hypotheses, goal)
        return goal

if __name__ == '__main__':
    text = """
    -- This is a comment
    /- This is a nested comment -/
    theorem foo : 1 = 1 := rfl
    lemma bar : 2 = 2 := rfl
    """
    print("Testing Lean3Utils")
    print("-"*20)
    print("Testing Lean3Utils.remove_comments")
    print("Before:")
    print(text)
    print("After:")
    print(Lean3Utils.remove_comments(text))
    print("-"*20)
    print("Testing Lean3Utils.get_lean_root_path")
    print(Lean3Utils.get_lean_root_path())
    print("-"*20)
    print("Testing Lean3Utils.get_lean_lib_path")
    print(Lean3Utils.get_lean_lib_path())
    print("-"*20)
    print("Testing Lean3Utils.find_theorems")
    lib_path = Lean3Utils.get_lean_lib_path()
    init_data_name = "init/data/array/basic.lean"
    init_data_path = os.path.join(lib_path, init_data_name)
    with open(init_data_path, 'r') as f:
        text = f.read()
    print(f"Testing Lean3Utils.find_theorems on {init_data_name}")
    for name, dfn in Lean3Utils.find_theorems(text):
        print(f"{name}: {dfn}")
    print("-"*20)
    print("Testing Lean3Utils.find_theorems_with_namespaces")
    for namespace, name, dfn in Lean3Utils.find_theorems_with_namespaces(text):
        print(f"[{namespace}]::{name}: {dfn}")
    print("-"*20)
    # print(text)

    # print(Lean3Utils.find_theorems(text))
    # print(Lean3Utils.find_theorems_with_namespaces(text))