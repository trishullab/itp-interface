#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import os
import logging
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from collections import OrderedDict
from typing import List, Optional, Union, runtime_checkable, Protocol
from pydantic import BaseModel
from itp_interface.tools.tactic_parser import DeclWithDependencies

@runtime_checkable
class TrainingDataFormat(Protocol):
    
    def to_json(self, indent=0) -> str:
        raise NotImplementedError("to_json must be implemented by the child class")

    @staticmethod
    def load_from_file(file_path: str):
        raise NotImplementedError("load_from_file must be implemented by the child class")
    
    @staticmethod
    def load_from_string(json_text: str):
        raise NotImplementedError("load_from_string must be implemented by the child class")


@runtime_checkable
class MergableCollection(Protocol):
    def merge(self, __o: object):
        raise NotImplementedError("merge must be implemented by the child class")
    
    def undo_merge(self, size: int = 1, start_idx = 0) -> object:
        raise NotImplementedError("undo_merge must be implemented by the child class")

    def __len__(self) -> int:
        raise NotImplementedError("__len__ must be implemented by the child class")


@runtime_checkable
class TrainingDataCollection(Protocol):
    training_data: list

    def merge(self, __o: object):
        raise NotImplementedError("merge must be implemented by the child class")
    
    def undo_merge(self, size: int = 1, start_idx = 0) -> object:
        raise NotImplementedError("undo_merge must be implemented by the child class")

    def __len__(self) -> int:
        return len(self.training_data)
    
    def to_json(self, indent=0) -> str:
        raise NotImplementedError("to_json must be implemented by the child class")

    @staticmethod
    def load_from_file(file_path: str, logger: logging.Logger = None):
        raise NotImplementedError("load_from_file must be implemented by the child class")

    @staticmethod
    def load_from_string(json_text: str, logger: logging.Logger = None):
        raise NotImplementedError("load_from_string must be implemented by the child class")


@dataclass_json
@dataclass
class LemmaRefWithScore(object):
    """Class to store the lemma reference with score."""
    lemma_idx: int
    score: float
    pass

@dataclass_json
@dataclass
class Goal(object):
    """Class to store the goal."""
    hypotheses: List[str] = field(default_factory=list) # The list of hypothesis for the goal.
    goal: Optional[str] = None
    relevant_defns: List[LemmaRefWithScore] = field(default_factory=list) # The list of relevant definitions.
    used_theorems_local: List[LemmaRefWithScore] = field(default_factory=list) # The list of useful theorems.
    used_theorems_external: List[LemmaRefWithScore] = field(default_factory=list) # The list of useful theorems.
    possible_useful_theorems_external: List[LemmaRefWithScore] = field(default_factory=list) # The list of possible useful theorems.
    possible_useful_theorems_local: List[LemmaRefWithScore] = field(default_factory=list) # The list of possible useful theorems.

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Goal):
            return False
        if self.goal != __o.goal:
            return False
        hyp1_set = set(self.hypotheses)
        hyp2_set = set(__o.hypotheses)
        return hyp1_set.difference(hyp2_set) == set() and hyp2_set.difference(hyp1_set) == set() and len(self.hypotheses) == len(__o.hypotheses)
    
    def __le__(self, __o: object) -> bool:
        # To goal 'a' is less (hard) than goal 'b' iff all hypotheses of 'b' are also hypotheses of 'a'
        if not isinstance(__o, Goal):
            raise TypeError(f"Cannot compare Goal with {type(__o)}")
        if self.goal != __o.goal:
            raise ValueError(f"Cannot compare goals with different goals: {self.goal} != {__o.goal}")
        set_a = set(self.hypotheses)
        set_b = set(__o.hypotheses)
        b_is_subset_of_a = set_b.issubset(set_a)
        return b_is_subset_of_a
    
    def __ge__(self, __o: object) -> bool:
        # To goal 'a' is more (hard) than goal 'b' iff all hypotheses of 'a' are also hypotheses of 'b'
        if not isinstance(__o, Goal):
            raise TypeError(f"Cannot compare Goal with {type(__o)}")
        if self.goal != __o.goal:
            raise ValueError(f"Cannot compare goals with different goals: {self.goal} != {__o.goal}")
        set_a = set(self.hypotheses)
        set_b = set(__o.hypotheses)
        a_is_subset_of_b = set_a.issubset(set_b)
        return a_is_subset_of_b
    
    def __lt__(self, __o: object) -> bool:
        return self != __o and self <= __o 
    
    def __gt__(self, __o: object) -> bool:
        return self != __o and self >= __o
    
    def to_json(self, indent=0) -> str:
        return Goal.schema().dumps(self, indent=indent)

    @staticmethod
    def load_from_file(file_path: str):
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return Goal.load_from_string(json_text)
    
    @staticmethod
    def load_from_string(json_text: str):
        assert json_text is not None, "json_text cannot be None"
        return Goal.schema().loads(json_text)

@dataclass_json
@dataclass
class LemmaReferences(object):
    """Class to store the lemma references."""
    lemma_idx: int
    lemma_name: str
    lemma_defn: str
    ref_count: int = 0
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, LemmaReferences):
            return False
        return self.lemma_name == __o.lemma_name and self.lemma_defn == __o.lemma_defn
    
    def __hash__(self) -> int:
        return hash((self.lemma_name, self.lemma_defn))
    
    def __str__(self) -> str:
        return f"{self.lemma_name} : {self.lemma_defn}"
#        return f"{self.lemma_defn} : {self.lemma_name}"

    def clone(self, idx : Optional[int] = None):
        new_copy = copy.deepcopy(self)
        if idx is not None:
            new_copy.lemma_idx = idx
        return new_copy
    
    def to_json(self, indent=0) -> str:
        return LemmaReferences.schema().dumps(self, indent=indent)
    
    @staticmethod
    def load_from_file(file_path: str):
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return LemmaReferences.load_from_string(json_text)
    
    @staticmethod
    def load_from_string(json_text: str):
        assert json_text is not None, "json_text cannot be None"
        return LemmaReferences.schema().loads(json_text)

@dataclass_json
@dataclass
class TheoremProvingTrainingDataFormat(object):
    """Class to format the training data for coq based automatic theorem provers.
    This class is responsible for formatting the training data for coq based automatic theorem provers.
    """
    proof_id : Optional[str] = None # The id of the proof which helps locating the proof in the original file.    
    all_useful_defns_theorems : List[LemmaReferences] = field(default_factory=list) # The list of all useful definitions.
    goal_description: Optional[str] = None # The description of the goal.
    start_goals: List[Goal] = field(default_factory=list) # The goal to start with.
    end_goals: List[Goal] = field(default_factory=list) # The goal to end with.
    proof_steps: List[str] = field(default_factory=list) # The list of proof steps to get from the start goal to the end goal.
    simplified_goals: List[Goal] = field(default_factory=list) # A possible list of simplified theorem or lemma to prove.
    addition_state_info: dict = field(default_factory=dict) # Custom key-value pairs for additional information.
    file_path: Optional[str] = None # The path of the file which contains the proof.
    project_id: Optional[str] = None # The url of the repository which contains the proof.
    theorem_name: Optional[str] = None # The name of the theorem.

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, TheoremProvingTrainingDataFormat):
            return False
        goal_set_a = set([goal.goal for goal in self.start_goals])
        goal_set_b = set([goal.goal for goal in __o.start_goals])
        if goal_set_a.difference(goal_set_b) != set() or goal_set_b.difference(goal_set_a) != set():
            return False
        goals_a = OrderedDict()
        goals_b = OrderedDict()
        for goal in self.start_goals:
            if goal.goal not in goals_a:
                goals_a[goal.goal] = [goal]
            else:
                goals_a[goal.goal].append(goal)
            goals_b[goal.goal] = []

        for goal in __o.start_goals:
            if goal.goal in goals_b:
                goals_b[goal.goal].append(goal)
        
        # Assert that goal_a keys are exactly the same as goal_b keys
        assert set(goals_a.keys()) == set(goals_b.keys()), "keys of goals_a and goals_b must be exactly the same"

        for key in goals_a:
            if len(goals_a[key]) != len(goals_b[key]):
                return False
            for g_a in goals_a[key]:
                if g_a not in goals_b[key]:
                    return False
            for g_b in goals_b[key]:
                if g_b not in goals_a[key]:
                    return False
        return True
                
        # # Create new goals with combined hypotheses
        # goals_a = [Goal(list([h for goal in value for h in goal.hypotheses]), key) for key, value in goals_a.items()]
        # goals_b = [Goal(list([h for goal in value for h in goal.hypotheses]), key) for key, value in goals_b.items()]
        # return all([goal_a == goal_b for goal_a, goal_b in zip(goals_a, goals_b)])
    
    def __le__(self, __o: object) -> bool:
        # TrainingDataFormat 'a' is less (hard) than TrainingDataFormat 'b' iff all goals in 'a' are subset of goals in 'b'
        if not isinstance(__o, TheoremProvingTrainingDataFormat):
            raise TypeError(f"Cannot compare TrainingDataFormat with {type(__o)}")
        goal_set_a = set([goal.goal for goal in self.start_goals])
        goal_set_b = set([goal.goal for goal in __o.start_goals])
        a_is_subset_of_b = goal_set_a <= goal_set_b
        if not a_is_subset_of_b:
            return False
        else:
            # Go over all subset goals
            # Check if each goal is as hard as the goal in __o
            # Combine all hypotheses for same goal
            goals_a = OrderedDict()
            goals_b = OrderedDict()
            for goal in self.start_goals:
                if goal.goal not in goals_a:
                    goals_a[goal.goal] = [goal]
                else:
                    goals_a[goal.goal].append(goal)
                goals_b[goal.goal] = []

            for goal in __o.start_goals:
                if goal.goal in goals_b:
                    goals_b[goal.goal].append(goal)

            for key in goals_a:
                # Clearly goals in a are present in goals in b
                if len(goals_a[key]) > len(goals_b[key]):
                    return False
            # For all the goals in a which are matching b goals
            # The number of such goals in a is always less than that of b
            # So b is harder as we have more goals to prove
            for key in goals_a:
                for g_a in goals_a[key]:
                    for g_b in goals_b[key]:
                        a_is_strictly_harder_than_b = g_a > g_b and g_b < g_a # g_a > g_b is not same as g_b < g_a as because it is not a total order and the goals can be incomparable
                        a_is_not_comparable_to_b = (not g_a > g_b) and (not g_b < g_a)
                        if a_is_not_comparable_to_b or a_is_strictly_harder_than_b:
                            return False
            return True
            # for key in goals_a:
            #     for g_a in goals_a[key]:
            #         if g_a not in goals_b[key]:
            #             return False
            #     for g_b in goals_b[key]:
            #         if g_b not in goals_a[key]:
            #             return False
            
            # # Create new goals with combined hypotheses
            # goals_a = [Goal(list([h for goal in value for h in goal.hypotheses]), key) for key, value in goals_a.items()]
            # goals_b = [Goal(list([h for goal in value for h in goal.hypotheses]), key) for key, value in goals_b.items()]
 
            # a_less_harder_than_b = all([g_a <= g_b for g_a, g_b in zip(goals_a, goals_b)])
            # return a_less_harder_than_b
    
    def __ge__(self, __o: object) -> bool:
        # TrainingDataFormat 'a' is more (hard) than TrainingDataFormat 'b' iff all goals in 'b' are subset of goals in 'a'
        if not isinstance(__o, TheoremProvingTrainingDataFormat):
            raise TypeError(f"Cannot compare TrainingDataFormat with {type(__o)}")
        goal_set_a = set([goal.goal for goal in self.start_goals])
        goal_set_b = set([goal.goal for goal in __o.start_goals])
        b_is_subset_of_a = goal_set_b <= goal_set_a
        if not b_is_subset_of_a:
            return False
        else:
            # Go over all subset goals
            # Check if each goal is as hard as the goal in __o
            goals_a = OrderedDict()
            goals_b = OrderedDict()
            for goal in self.start_goals:
                if goal.goal not in goals_a:
                    goals_a[goal.goal] = [goal]
                else:
                    goals_a[goal.goal].append(goal)
                goals_b[goal.goal] = []

            for goal in __o.start_goals:
                if goal.goal in goals_b:
                    goals_b[goal.goal].append(goal)

            for key in goals_b:
                # Clearly goals in a are present in goals in b
                if len(goals_a[key]) < len(goals_b[key]):
                    return False
            # For all the goals in a which are matching b goals
            # The number of such goals in a is always less than that of b
            # So b is harder as we have more goals to prove
            for key in goals_b:
                for g_b in goals_b[key]:
                    for g_a in goals_a[key]:
                        b_is_strictly_harder_than_a = g_a < g_b and g_b > g_a # g_a < g_b is not same as g_b > g_a as because it is not a total order and the goals can be incomparable
                        b_is_not_comparable_to_a = (not g_a < g_b) and (not g_b > g_a)
                        if b_is_not_comparable_to_a or b_is_strictly_harder_than_a:
                            return False
            return True

            # goals_a = [Goal(list([h for goal in value for h in goal.hypotheses]), key) for key, value in goals_a.items()]
            # goals_b = [Goal(list([h for goal in value for h in goal.hypotheses]), key) for key, value in goals_b.items()]

            # b_less_harder_than_a = all([g_a >= g_b for g_a, g_b in zip(goals_a, goals_b)])
            # return b_less_harder_than_a
    
    def __lt__(self, __o: object) -> bool:
        return self != __o and self <= __o
    
    def __gt__(self, __o: object) -> bool:
        return self != __o and self >= __o

    def __hash__(self) -> int:
        goal_set = list(set([goal.goal for goal in self.start_goals]))
        goal_set.sort()
        return hash(tuple(goal_set))

    def have_same_proof_steps(self, __o: object) -> bool:
        if not isinstance(__o, TheoremProvingTrainingDataFormat):
            raise TypeError(f"Cannot compare TrainingDataFormat with {type(__o)}")
        return len(self.proof_steps) == len(__o.proof_steps) and all([p_a == p_b for p_a, p_b in zip(self.proof_steps, __o.proof_steps)])

    def get_human_readable_serialized_goal(self, idx: int, skip_special_tokens: bool = False):
        assert idx >= 0 and idx < len(self.start_goals), f"idx must be in range [0, {len(self.start_goals)})"
        hyps = "\n".join(self.start_goals[idx].hypotheses)
        return f"""{f"Goal {idx + 1}:" if not skip_special_tokens else ""}
{self.start_goals[idx].goal}
{f"Hyps {idx + 1}:" if not skip_special_tokens else ""}
{hyps}
"""

    def to_json(self, indent=0) -> str:
        return TheoremProvingTrainingDataFormat.schema().dumps(self, indent=indent)

    @staticmethod
    def load_from_file(file_path: str):
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return TheoremProvingTrainingDataFormat.load_from_string(json_text)
    
    @staticmethod
    def load_from_string(json_text: str):
        assert json_text is not None, "json_text cannot be None"
        return TheoremProvingTrainingDataFormat.schema().loads(json_text)    

@dataclass_json
@dataclass
class LemmaReferencesCollection(TrainingDataCollection):
    """Class to store the lemma references."""
    training_data: list[LemmaReferences] = field(default_factory=list)
    
    def __post_init__(self):
        self._lemma_ref_to_idx = {lemma_ref: idx for idx, lemma_ref in enumerate(self.training_data)}

    def merge(self, __o: object):
        """
        Merge the lemma references with another lemma references collection.
        Returns the merged lemma references collection index map.
        """
        if not isinstance(__o, LemmaReferencesCollection) and not isinstance(__o, LemmaReferences) and not isinstance(__o, list):
            raise TypeError(f"Cannot merge LemmaReferenceCollection with {type(__o)}")
        if isinstance(__o, list) and not all(isinstance(x, LemmaReferences) for x in __o):
            raise TypeError(f"Cannot merge LemmaReferenceCollection with list of {type(__o)}")
        if isinstance(__o, LemmaReferences):
            __o = [__o]
        elif isinstance(__o, LemmaReferencesCollection):
            __o = __o.training_data
        to_take_cnt = len(__o)
        new_idx_map = [-1] * to_take_cnt
        for idx in range(to_take_cnt):
            lemma_ref = __o[idx]
            assert 0 <= lemma_ref.lemma_idx < len(__o), f"lemma_idx must be in range [0, {len(__o)}"
            if lemma_ref not in self._lemma_ref_to_idx:
                self._lemma_ref_to_idx[lemma_ref] = len(self.training_data)
                lemma_ref_copy = copy.deepcopy(lemma_ref)
                lemma_ref_copy.lemma_idx = len(self.training_data)
                self.training_data.append(lemma_ref_copy)
                new_idx_map[idx] = lemma_ref_copy.lemma_idx
            else:
                lemma_idx = self._lemma_ref_to_idx[lemma_ref]
                new_idx_map[idx] = lemma_idx
                self.training_data[lemma_idx].ref_count += lemma_ref.ref_count
        assert all(idx != -1 for idx in new_idx_map), "new_idx_map must not contain any -1 values"
        return new_idx_map
    
    def undo_merge(self, size: int = 1, start_idx=0) -> object:
        assert size >= 1, "size must be greater than equal to 1"
        assert start_idx >= 0, "start_idx must be greater than zero"
        assert start_idx < len(self.training_data), f"can only cut-down from idx < {len(self.training_data)}"
        fraction = self.training_data[start_idx: size]
        return LemmaReferencesCollection(training_data=fraction)
    
    def __iter__(self):
        return iter(self.training_data)
    
    def __getitem__(self, idx: int) -> LemmaReferences:
        return self.training_data[idx]

    def to_json(self, indent=0) -> str:
        return LemmaReferencesCollection.schema().dumps(self, indent=indent)

    @staticmethod
    def load_from_file(file_path: str, logger: logging.Logger = None):
        assert os.path.exists(file_path), f"file_path:{file_path} must be a valid path to a file"
        json_text = None
        if logger is not None:
            logger.info(f"Loading json data from {file_path}")
        with open(file_path, "r") as f:
            json_text = f.read()
        if logger is not None:
            logger.info(f"Loaded json data from {file_path}")
        return LemmaReferencesCollection.load_from_string(json_text, logger)

    @staticmethod
    def load_from_string(json_text: str, logger: logging.Logger = None):
        assert json_text is not None, "json_text cannot be None"
        if logger is not None:
            logger.info(f"Deserializing json data from string of length {len(json_text)} characters")
        deserialized = LemmaReferencesCollection.schema().loads(json_text)
        if logger is not None:
            logger.info(f"Deserialized json data from string of length {len(json_text)} characters")
        return deserialized

@dataclass_json
@dataclass
class TheoremProvingTrainingDataCollection(TrainingDataCollection):
    training_data: list[TheoremProvingTrainingDataFormat] = field(default_factory=list) # The list of training data.

    def merge(self, __o: object):
        assert isinstance(__o, TheoremProvingTrainingDataCollection)
        self.training_data.extend(__o.training_data)
    
    def undo_merge(self, size: int = 1, start_idx=0) -> object:
        assert size >= 1, "size must be greater than equal to 1"
        assert start_idx >= 0, "start_idx must be greater than zero"
        assert start_idx < len(self.training_data), f"can only cut-down from idx < {len(self.training_data)}"
        fraction = self.training_data[start_idx: size]
        return TheoremProvingTrainingDataCollection(training_data=fraction)

    def to_json(self, indent=0) -> str:
        return TheoremProvingTrainingDataCollection.schema().dumps(self, indent=indent)

    @staticmethod
    def load_from_file(file_path: str, logger: logging.Logger = None):
        assert os.path.exists(file_path), f"file_path: {file_path} must be a valid path to a file"
        json_text = None
        if logger is not None:
            logger.info(f"Loading json data from {file_path}")
        with open(file_path, "r") as f:
            json_text = f.read()
        if logger is not None:
            logger.info(f"Loaded json data from {file_path}")
        return TheoremProvingTrainingDataCollection.load_from_string(json_text, logger)

    @staticmethod
    def load_from_string(json_text: str, logger: logging.Logger = None):
        assert json_text is not None, "json_text cannot be None"
        if logger is not None:
            logger.info(f"Deserializing json data from string of length {len(json_text)} characters")
        deserialized = TheoremProvingTrainingDataCollection.schema().loads(json_text)
        if logger is not None:
            logger.info(f"Deserialized json data from string of length {len(json_text)} characters")
        return deserialized

class ExtractionDataCollection(BaseModel):
    training_data: list[DeclWithDependencies] = []

    def __len__(self) -> int:
        return len(self.training_data)

    def to_json(self, indent=0) -> str:
        if indent == 0:
            return self.model_dump_json()
        else:
            return self.model_dump_json(indent=indent)

    def merge(self, __o: object):
        assert isinstance(__o, ExtractionDataCollection)
        self.training_data.extend(__o.training_data)
    
    def undo_merge(self, size: int = 1, start_idx=0) -> object:
        assert size >= 1, "size must be greater than equal to 1"
        assert start_idx >= 0, "start_idx must be greater than zero"
        assert start_idx < len(self.training_data), f"can only cut-down from idx < {len(self.training_data)}"
        fraction = self.training_data[start_idx: size]
        return ExtractionDataCollection(training_data=fraction)
    
    @staticmethod
    def load_from_string(json_text: str, logger: logging.Logger = None):
        assert json_text is not None, "json_text cannot be None"
        return ExtractionDataCollection.model_validate_json(json_text)

    @staticmethod
    def load_from_file(file_path: str, logger: logging.Logger = None):
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return ExtractionDataCollection.load_from_string(json_text, logger=logger)

@dataclass_json
@dataclass
class TrainingDataMetadataFormat(MergableCollection):
    """Class to store the training data metadata.

    This class is responsible for storing the training data metadata.
    """
    training_data_buffer_size: int = 10000
    last_training_data: int = 0
    last_proof_id: Optional[str] = None
    external_theorems_used_cnt: int = 0
    local_theorems_used_cnt: int = 0
    total_data_count: int = 0
    data_filename_prefix: str = "full_data"
    data_filename_suffix: str = ".json"
    lemma_ref_filename_prefix: str = "full_data_lemma_ref"
    lemma_ref_filename_suffix: str = ".json"
    num_theorems: int = 0

    def merge(self, __o: object):
        if not isinstance(__o, TrainingDataMetadataFormat):
            raise TypeError(f"Cannot merge TrainingDataMetadata with {type(__o)}")
        self.training_data_buffer_size = max(__o.training_data_buffer_size, self.training_data_buffer_size)
        self.last_training_data = __o.last_training_data
        self.last_proof_id = __o.last_proof_id
        self.total_data_count += __o.total_data_count
        self.external_theorems_used_cnt += __o.external_theorems_used_cnt
        self.local_theorems_used_cnt += __o.local_theorems_used_cnt
        self.num_theorems += __o.num_theorems
    
    def undo_merge(self, size: int = 1, start_idx=0) -> object:
        raise NotImplementedError("undo_merge is not implemented for TrainingDataMetadataFormat")
    
    def __len__(self) -> int:
        return 0

    def to_json(self, indent=0) -> str:
        return TrainingDataMetadataFormat.schema().dumps(self, indent=indent)

    @staticmethod
    def load_from_file(file_path: str):
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return TrainingDataMetadataFormat.load_from_string(json_text)

    @staticmethod
    def load_from_string(json_text: str):
        assert json_text is not None, "json_text cannot be None"
        return TrainingDataMetadataFormat.schema().loads(json_text)


class TrainingDataFormatLayout(object):
    def __init__(self, with_labels: bool = False):
        self.with_labels = with_labels
        pass
    
    def get_layout_format_name(self) -> str:
        raise NotImplementedError("get_layout_format_name must be implemented in derived classes")

    def layout_training_data(self, training_data_format: TheoremProvingTrainingDataFormat) -> Union[str, tuple[str, str]]:
        raise NotImplementedError("get_formatted_training_data must be implemented in derived classes")
    
    def get_training_data_from_layout(self, formatted_training_data: str) -> TheoremProvingTrainingDataFormat:
        raise NotImplementedError("get_training_data_format must be implemented in derived classes")
    
if __name__ == "__main__":
    # Test the training data collection
    training_data_format1 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['e : expr'], goal="size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['n : nat'], goal="1 <= 1"),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m'),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )

    training_data_format2 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['e : expr'], goal="size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['n : nat'], goal="1 <= 1"),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m'),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )

    training_data_format3 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['e : expr'], goal="size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m'),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )

    training_data_format4 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['e : expr'], goal="size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['n : nat'], goal="1 <= 1"),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )

    training_data_format5 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['e : expr'], goal="size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )

    training_data_format6 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )

    training_data_format7 = TheoremProvingTrainingDataFormat(
        proof_id="proof_id",
        start_goals=[
            Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e"),
            Goal(hypotheses=['v : var'], goal="1 <= 1"),
            Goal(hypotheses=['IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e3 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n '),
            Goal(hypotheses=['m : var_map', 'IHe2 : forall m : var_map, eval_expr e2 m = eval_expr (constant_fold e2) m', 'IHe1 : forall m : var_map, eval_expr e1 m = eval_expr (constant_fold e1) m', 'e1,e2 : expr'], goal='eval_expr e1 m + eval_expr e2 m =\neval_expr\n  match constant_fold e1 with\n  | Const (0 as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => constant_fold e2\n      end\n  | Const (S _ as n1) =>\n      match constant_fold e2 with\n      | Const n2 => Const (n1 + n2)\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  | _ =>\n      match constant_fold e2 with\n      | Const 0 => constant_fold e1\n      | _ => Plus (constant_fold e1) (constant_fold e2)\n      end\n  end m')
        ],
        end_goals=[],
        proof_steps=[],
        simplified_goals=[],
        addition_state_info={}
    )    
    assert Goal(hypotheses=[], goal="forall e : expr, size (constant_fold e) <= size e") >= Goal(hypotheses=['e: expr'], goal="forall e : expr, size (constant_fold e) <= size e")
    assert training_data_format1 <= training_data_format2 and training_data_format1 >= training_data_format2, "Training data format comparison failed"
    assert training_data_format2 <= training_data_format3
    assert training_data_format3 >= training_data_format2
    assert not (training_data_format1 >= training_data_format3)
    assert not (training_data_format3 <= training_data_format1)
    assert not (training_data_format3 <= training_data_format4)
    assert not (training_data_format4 >= training_data_format3)
    assert training_data_format5 >= training_data_format4 and training_data_format5 <= training_data_format3
    assert training_data_format6 <= training_data_format5 <= training_data_format3
    assert not (training_data_format7 <= training_data_format3) and not (training_data_format7 >= training_data_format3)