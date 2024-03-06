import art
import os


def terminal_header(text: str) -> None:
    """Prints a header to the terminal.

    Args:

        text: str
            The text to print as a header.

    """
    try:
        size = os.get_terminal_size()
        c = size.columns
        txt = "-" * c
        print("")
        print(txt)
        print(art.text2art(text))
    except:
        pass
