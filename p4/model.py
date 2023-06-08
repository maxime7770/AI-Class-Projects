import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.learning_rate = 0.1
        self.numTrainingGames = 10000
        self.w1 = nn.Parameter(state_dim, 128)
        self.w2 = nn.Parameter(128, 64)
        self.w3 = nn.Parameter(64, 32)
        self.w4 = nn.Parameter(32, 16)
        self.w5 = nn.Parameter(16, action_dim)
        self.b1 = nn.Parameter(1, 128)
        self.b2 = nn.Parameter(1, 64)
        self.b3 = nn.Parameter(1, 32)
        self.b4 = nn.Parameter(1, 16)
        self.b5 = nn.Parameter(1, action_dim)
        self.batch_size = 16

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        o1 = nn.ReLU(nn.AddBias(nn.Linear(states, self.w1), self.b1))
        o2 = nn.ReLU(nn.AddBias(nn.Linear(o1, self.w2), self.b2))
        o3 = nn.ReLU(nn.AddBias(nn.Linear(o2, self.w3), self.b3))
        o4 = nn.ReLU(nn.AddBias(nn.Linear(o3, self.w4), self.b4))
        o5 = nn.AddBias(nn.Linear(o4, self.w5), self.b5)
    
        return o5

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """

        loss = self.get_loss(states, Q_target)
        grads = nn.gradients(loss, [self.w1, self.w2, self.w3, self.w4, self.w5, self.b1, self.b2, self.b3, self.b4, self.b5])
        
        self.w1.update(grads[0], -self.learning_rate)
        self.w2.update(grads[1], -self.learning_rate)
        self.w3.update(grads[2], -self.learning_rate)
        self.w4.update(grads[3], -self.learning_rate)
        self.w5.update(grads[4], -self.learning_rate)
        self.b1.update(grads[5], -self.learning_rate)
        self.b2.update(grads[6], -self.learning_rate)
        self.b3.update(grads[7], -self.learning_rate)
        self.b4.update(grads[8], -self.learning_rate)
        self.b5.update(grads[9], -self.learning_rate)
