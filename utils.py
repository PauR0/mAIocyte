


def check_s2_params(s2_ini, s2_end, s2_step):
    """
    This function checks the relation between the three args so they fulfill:
    if s2_ini < s2_end => s2_step > 0, otherwise if s2_ini > s2_end => s2_step < 0.
    If the sign of s2_step is wrong, it is corrected. In addition it returns a delta
    function to check if a given s2 has reached s2_end.

    Arguments:
    -----------
        s2_ini  : float
        s2_end  : float
        s2_step : float

    Returns:
    ----------
        s2_step : float
        cond : lambda function

    """

    if s2_ini < s2_end:
        cond = lambda s2 : s2_end >= s2
        if s2_step < 0:
            print("Correcting s2_step sign")
            s2_step *= -1
    else:
        cond = lambda s2 : s2_end <= s2
        if s2_step > 0:
            print("Correcting s2_step sign")
            s2_step *= -1

    return s2_step, cond