import sys


def is_debug_mode() -> bool:
    """
    Determines if the current Python process is running under a debugger.

    This function checks for the presence of common debugger modules in sys.modules,
    such as those used by PyCharm ("pydevd"), Visual Studio Code ("ptvsd"), the built-in
    Python debugger ("pdb"), or IPython environments (which often indicate interactive debugging).

    Returns:
        bool: True if a known debugger module is loaded (i.e., debugging is likely active), False otherwise.
    """
    return any(mod in sys.modules for mod in ["pydevd", "ptvsd", "pdb", "IPython"])


if __name__ == "__main__":
    print("is_debug_mode():", is_debug_mode())
