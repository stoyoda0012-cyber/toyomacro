"""Print the AI-agent user guide of the installed version."""

import sys

from toyomacro.guides import read_guide

# The guide is UTF-8 and contains characters (em dashes, arrows) that a
# legacy console or redirect encoding such as cp932 or cp1252 cannot encode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print(read_guide(), end="")
