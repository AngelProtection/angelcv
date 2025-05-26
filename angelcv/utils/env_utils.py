import sys


def is_debug_mode() -> bool:
    """
    Check if the code is currently running in debug mode.

    This function detects whether a Python debugger (like pdb, PyCharm debugger,
    or VS Code debugger) is currently active by checking the sys.gettrace() function.

    Returns:
        bool: True if a debugger is active, False otherwise
    """
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        return False  # Debugging is not supported or not in debugger mode
    return gettrace() is not None  # Returns True if debugger is active, False otherwise
