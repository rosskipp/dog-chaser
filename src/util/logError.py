import traceback


def logError(e: Exception):
    print("Error calculating metric")
    print("---------")
    print(e)
    print(e.__doc__)
    print("---------")
    print(traceback.format_exc())
    return
