from typing import List, Dict, Any, Type, NamedTuple

class Obligation(NamedTuple):
    hypotheses: List[str]
    goal: str

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"hypotheses": self.hypotheses,
                "goal": self.goal}
    
class ProofContext(NamedTuple):
    fg_goals: List[Obligation]

    @classmethod
    def empty(cls: Type['ProofContext']):
        return ProofContext([])

    @classmethod
    def from_dict(cls, data):
        fg_goals = list(map(Obligation.from_dict, data["fg_goals"]))
        return cls(fg_goals)

    def to_dict(self) -> Dict[str, Any]:
        return {"fg_goals": list(map(Obligation.to_dict, self.fg_goals))}

    @property
    def all_goals(self) -> List[Obligation]:
        return self.fg_goals

    @property
    def focused_goal(self) -> str:
        if self.fg_goals:
            return self.fg_goals[0].goal
        else:
            return ""

    @property
    def focused_hyps(self) -> List[str]:
        if self.fg_goals:
            return self.fg_goals[0].hypotheses
        else:
            return []