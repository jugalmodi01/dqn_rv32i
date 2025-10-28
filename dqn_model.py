import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# Define a named tuple for our experience replay
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """Experience replay memory to store and sample transitions"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """Agent that uses DQN to learn and select actions"""
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Memory parameters
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-networks (current and target)
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Track number of learning steps
        self.learn_step_counter = 0
        
        # Used instructions to avoid redundancy (track operation types, not full instructions)
        self.used_instructions = set()
        
        # For metrics tracking
        self.loss_history = []
        self.reward_history = [] # to store total reward per episode
        self.epsilon_history = []
        self.current_episode_reward = 0
        
    def select_action(self, state, instruction_generator, deterministic=False):
        """Select an action using epsilon-greedy policy"""
        if deterministic or random.random() > self.epsilon:
            # Exploit: choose the best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                # Get top actions by Q-value (limit search to top 10 to avoid unnecessary iterations)
                top_k = min(10, self.action_size)
                top_actions = torch.topk(q_values, top_k, dim=1)[1].squeeze()
                
                # Convert to list if it's a single value
                if top_actions.dim() == 0:
                    top_actions = [top_actions.item()]
                else:
                    top_actions = top_actions.tolist()
                
                # Try top actions first
                for action_idx in top_actions:
                    # Use operation type as a proxy to check if similar instructions were used
                    # This is much faster than generating and checking every instruction
                    operation = instruction_generator.operation_list[action_idx]
                    if operation not in self.used_instructions:
                        self.used_instructions.add(operation)
                        return action_idx
                
                # If all top actions are used, fall back to random
                return self.random_unused_action(instruction_generator)
        else:
            # Explore: choose a random action
            return self.random_unused_action(instruction_generator)
    
    def random_unused_action(self, instruction_generator):
        """Choose a random action that hasn't been used before"""
        # Use operation type tracking instead of full instruction generation
        # This is much faster as we don't need to generate instructions
        unused_actions = [i for i in range(self.action_size) 
                         if instruction_generator.operation_list[i] not in self.used_instructions]
        
        if unused_actions:
            action_idx = random.choice(unused_actions)
            operation = instruction_generator.operation_list[action_idx]
            self.used_instructions.add(operation)
            return action_idx
        
        # If all operation types have been used, clear and start fresh
        self.used_instructions.clear()
        action_idx = random.randrange(self.action_size)
        operation = instruction_generator.operation_list[action_idx]
        self.used_instructions.add(operation)
        return action_idx

    def end_episode(self):
        """Should be called at the end of each episode to update metrics."""
        self.reward_history.append(self.current_episode_reward)
        self.current_episode_reward = 0
        self.used_instructions.clear()
    
    def update_epsilon(self):
        """Update epsilon according to decay schedule"""
        self.epsilon_history.append(self.epsilon)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, next_state, reward, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, next_state, reward, done)
        self.current_episode_reward += reward
    
    def learn(self, target_update=10):
        """Update the policy network based on stored experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors and move to device
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Calculate current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Calculate target Q values using Double DQN approach for better stability
        with torch.no_grad():
            # Use policy network to select actions, target network to evaluate them
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.loss_history.append(loss.item())
            
        return loss.item()