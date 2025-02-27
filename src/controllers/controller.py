# controller.py

def update_parameter(param, grad, learning_rate):
    """
    Perform a simple gradient descent update for a parameter.

    Parameters:
        param (float): Current value of the parameter.
        grad (float): Computed gradient of the loss with respect to the parameter.
        learning_rate (float): Step size for the gradient update.
    
    Returns:
        float: Updated parameter value.
    """
    return param - learning_rate * grad

def update_parameters_dict(params, grads, learning_rate):
    """
    Update a dictionary of parameters using gradient descent.

    Parameters:
        params (dict): Dictionary containing current parameter values.
        grads (dict): Dictionary containing gradients for each parameter.
        learning_rate (float): Step size for the update.
    
    Returns:
        dict: Dictionary with updated parameter values.
    """
    updated = {}
    for key in params:
        updated[key] = params[key] - learning_rate * grads.get(key, 0)
    return updated

# src/controllers/controller.py

def compute_gradients(P_next, target, fees_coefs, rewards_coefs):
    """
    Compute gradients of the loss L = (P_next - target)^2 with respect to fee/reward parameters.
    """
    error = P_next - target
    grad_fees = {key: 2 * error * fees_coefs.get(key, 0) for key in fees_coefs}
    grad_rewards = {key: 2 * error * rewards_coefs.get(key, 0) for key in rewards_coefs}
    return grad_fees, grad_rewards


class PIDController:
    """
    A Proportional-Integral-Derivative (PID) controller is a feedback control mechanism widely used in control systems.

    It computes a control signal based on three components:
    
        1. Proportional (P): Responds proportionally to the current error.
        2. Integral (I): Responds based on the accumulation of past errors.
        3. Derivative (D): Responds to the rate of change of the error.
    
    The control signal is computed as:
        output = Kp * error + Ki * integral + Kd * derivative

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        prev_error (float): The error from the previous time step (used to compute the derivative).
        integral (float): The accumulated sum of errors (used for the integral term).
    """

    def __init__(self, Kp, Ki, Kd):
        """
        Initialize the PID controller with specified gains.

        Parameters:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error):
        """
        Update the PID controller with the current error and compute the control signal.

        The update performs the following steps:
            1. Accumulate the error in the integral term.
            2. Compute the derivative as the difference between the current and previous errors.
            3. Compute the control output using the formula:
                   output = Kp * error + Ki * integral + Kd * derivative
            4. Store the current error as previous error for the next update.

        Parameters:
            error (float): The current error (difference between predicted and target price).

        Returns:
            float: The control signal that will be used to adjust the parameters.
        """
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
