import sys


def menu():
    message = '''
    ----------------------------------------------------------------------------------------
                            MENU: CHOOSE ONE OF THE FOLLOWING OPTIONS
    ----------------------------------------------------------------------------------------
    
    1   Run fire on chosen tessellation and generate gif
    2   Calculate propagation time graph as a function of p
    3   Determine the percolation threshold P_c
    4   Find the critical exponent for infinite cluster
    5   Compare probability bond vs probability site
    '''
    
    try:
        usrChoice = int(input(message))
        return usrChoice
    
    except ValueError:
        sys.exit('Exiting... The input was not an integer. Run and try again.')
     