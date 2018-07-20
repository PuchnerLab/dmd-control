from tkinter import Tk
from tkinter.filedialog import askopenfilename

"""
Graphical file handling functions. ``tkinter`` is used because it comes
standard with Python.
"""


def filepicker():
    """
    File-picker for loading a new transformation matrix in the middle
    of a session, just for convenience.
    """
    Tk().withdraw()
    filename = askopenfilename()
    # ``askopenfilename`` returns an empty ``tuple`` if no file is
    # selected. Return an empty ``str`` instead for type-safety.
    return '' if type(filename) == tuple else filename
