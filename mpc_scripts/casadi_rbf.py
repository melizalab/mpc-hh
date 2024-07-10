import casadi as ca

'''
Converts the embedded states, RBFN centers, and RBFN scale to a CasADI object
that do-mpc can use to make predictions.
'''

def PSI(state, centers, scale):
    # Define the other vectors as constants
    centers_sym = [ca.SX(center) for center in centers]

    # Calculate L2 distances
    RBFs = []
    for other_vec in centers_sym:
        distance = ca.dot((other_vec - state),(other_vec-state))
        RBFs.append(ca.exp(-scale * distance))

    # Create a function to evaluate the distances
    distances_function = ca.Function('distances_function', [state], RBFs)

    # Evaluate the distances
    distances = distances_function(state)

    return distances