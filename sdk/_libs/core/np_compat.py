import numpy as np

def _ensure(name, value):
    if name not in np.__dict__:
        setattr(np, name, value)

_ensure("int", int)
_ensure("float", float)
_ensure("bool", bool)
_ensure("object", object)
_ensure("complex", complex)
