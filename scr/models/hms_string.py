# Author: Fanli Zhou
# Date: 2020-06-10
#
# This script defines hms_string

def hms_string(sec_elapsed):
    """
    Returns the formatted time 

    Parameters:
    -----------
    sec_elapsed: int
        second elapsed

    Return:
    --------
    str
        the formatted time
    """

    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"