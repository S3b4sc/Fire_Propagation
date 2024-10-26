import sys


def menu():
    message = '''
    ----------------------------------------------------------------------------------------
                            MENU: CHOOSE ONE OF THE FOLLOWING OPTIONS
    ----------------------------------------------------------------------------------------
    
    1   Run squared Fire and generate gif
    2   Calculate propagation time graph as a function of p
    3   Determine the percolation threshold P_c
    
    '''
    
    try:
        usrChoice = int(input(message))
        return usrChoice
    
    except ValueError:
        sys.exit('Exiting... The input was not an integer. Run and try again.')
     