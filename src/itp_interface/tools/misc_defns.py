from enum import Enum

class HammerMode(Enum):
    NONE = 'NONE' # Prohibit hammer
    ALLOW = 'ALLOW' # Allow agent to query hammer
    AUTO = 'AUTO' # Automatically apply hammer (heuristically)
    ONESHOT = 'ONESHOT' # Proof attempt in one shot using hammer
    ALWAYS = 'ALWAYS' # Always use hammer after a successful proof attempt

    def __str__(self):
        return self.name