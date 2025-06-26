import sys


def is_debug_mode() -> bool:
    """
    Determines if the current Python process is running under a debugger.

    This function checks if a debugger is actually active, not just if debugger
    modules are imported (since many libraries like PyTorch import pdb as a dependency).

    Returns:
        bool: True if a debugger is actually active, False otherwise.
    """
    # Check if we're running under PyCharm debugger
    if "pydevd" in sys.modules:
        try:
            import pydevd  # type: ignore # noqa: I001

            return pydevd.GetGlobalDebugger() is not None
        except (ImportError, AttributeError):
            pass

    # Check if we're running under VS Code debugger
    if "ptvsd" in sys.modules or "debugpy" in sys.modules:
        return True

    # Check if pdb is actively debugging (not just imported)
    if "pdb" in sys.modules:
        try:
            import pdb

            # Check if there's an active pdb instance
            return hasattr(pdb, "Pdb") and pdb.Pdb._current_pdb is not None
        except (ImportError, AttributeError):
            pass

    # Check if we're in an IPython environment with debugger
    if "IPython" in sys.modules:
        try:
            from IPython import get_ipython  # type: ignore # noqa: I001

            ipython = get_ipython()
            if ipython is not None:
                # Check if IPython debugger is active
                return hasattr(ipython, "debugger") and ipython.debugger is not None
        except (ImportError, AttributeError):
            pass

    # Check if we're running with python -m pdb
    return sys.gettrace() is not None


if __name__ == "__main__":
    print("is_debug_mode():", is_debug_mode())
