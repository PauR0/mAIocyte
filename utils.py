


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

def check_attrs(obj, attr_names, err_message=None):
    """Check if given obj has the attributes in
    attr_names or id defined check if they are None.

    Arguments:
    ----------
        obj : any
            The object to be checked

        attr_names : list[str]
            The list of the names of the attributes to check.

        err_message : str, optional
            The error message to display. At the end of the message
            the followin text will be appedned:
                " {attr} has not been set."


    """

    for attr in attr_names:
            if not hasattr(obj, attr):
                if err_message is not None:
                    print(f"{err_message} \n\t {attr} has not been set")
                return
            elif getattr(obj, attr) is None:
                if err_message is not None:
                    print(f"{err_message}, \n\t{attr} has not been set")
                return

    return True
#