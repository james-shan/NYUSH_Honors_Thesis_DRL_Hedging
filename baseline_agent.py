import numpy as np

class DeltaNeutralAgent:
    def __init__(self, max_holding):
        self.max_holding = max_holding

    def act(self, state):
        """
        Generate an action aimed at achieving delta neutrality.
        
        Parameters:
        - state: The current state of the environment, expected to be a numpy array.
                 For 'PL' mode: state[0] = current holdings, state[4] = current option delta.
        
        Returns:
        - action: The action to take. This will be a desired change in holdings, scaled to [-1, 1].
        """
        # Extract relevant information from the state
        current_holdings = state[2]
        option_delta = state[4]

        # The target is to make the portfolio delta = 0
        # Since stock delta = 1 for each unit held, and option delta is provided,
        # the change in holdings required can be directly calculated from option_delta.
        
        # Calculate the desired total holdings to offset option delta
        desired_total_holdings = -option_delta*100  # For delta neutrality
        
        # Calculate the change required in current holdings to achieve this
        change_holdings = desired_total_holdings - current_holdings
        
        # Since actions are scaled to [-1, 1], we scale the delta_holdings accordingly
        action = np.clip(change_holdings / self.max_holding, -1, 1)
        
        return np.array([action], dtype=np.float32)
