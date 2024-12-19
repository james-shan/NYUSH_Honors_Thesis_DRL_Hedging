import numpy as np

class DeltaNeutralAgent:
    def __init__(self):
        pass

    def act(self, state, delta):
        """
        Generate an action aimed at achieving delta neutrality.
        
        Parameters:
        - state: The current state of the environment, expected to be a numpy array.
                 For 'PL' mode: state[0] = current holdings, delta = current option delta 
        
        Returns:
        - action: The action to take. This will be a desired change in holdings, scaled to [-1, 1].
        """
        # Extract relevant information from the state
        current_holdings = state[0]
        option_delta = delta

        # The target is to make the portfolio delta = 0
        # Since stock delta = 1 for each unit held, and option delta is provided,
        # the change in holdings required can be directly calculated from option_delta.
        
        # Calculate the desired total holdings to offset option delta
        desired_total_holdings = round(option_delta,3)  # For delta neutrality
        
        action = desired_total_holdings
        change_holdings = action-desired_total_holdings
        
        return np.array([action], dtype=np.float32)
