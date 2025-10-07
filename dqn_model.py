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
        
        # Initialize Q-networks (current and target)
        self.policy_net = DQN(state_size, action_size, hidden_size)
        self.target_net = DQN(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Track number of learning steps
        self.learn_step_counter = 0
        
        # Used instructions to avoid redundancy
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
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                
                # Sort actions by Q-value and try them in order until we find one that hasn't been used
                sorted_actions = torch.argsort(q_values, dim=1, descending=True).squeeze().numpy()
                
                for action_idx in sorted_actions:
                    # Get the operation name
                    operation = instruction_generator.operation_list[action_idx]
                    # Generate the instruction
                    instruction, _ = instruction_generator.generate_instruction(operation)
                    
                    # Check if this instruction has been used before
                    instr_hex = instruction_generator.format_instruction_hex(instruction)
                    if instr_hex not in self.used_instructions:
                        self.used_instructions.add(instr_hex)
                        return action_idx
                
                # If all preferred actions are used, fall back to random
                return self.random_unused_action(instruction_generator)
        else:
            # Explore: choose a random action
            return self.random_unused_action(instruction_generator)
    
    def random_unused_action(self, instruction_generator):
        """Choose a random action that hasn't been used before"""
        # Shuffle the action indices
        action_indices = list(range(self.action_size))
        random.shuffle(action_indices)
        
        for action_idx in action_indices:
            # Get the operation name
            operation = instruction_generator.operation_list[action_idx]
            # Generate the instruction
            instruction, _ = instruction_generator.generate_instruction(operation)
            
            # Check if this instruction has been used before
            instr_hex = instruction_generator.format_instruction_hex(instruction)
            if instr_hex not in self.used_instructions:
                self.used_instructions.add(instr_hex)
                return action_idx
        
        # If all actions have been used, clear the set and try again
        self.used_instructions.clear()
        action_idx = random.randrange(self.action_size)
        operation = instruction_generator.operation_list[action_idx]
        instruction, _ = instruction_generator.generate_instruction(operation)
        instr_hex = instruction_generator.format_instruction_hex(instruction)
        self.used_instructions.add(instr_hex)
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
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1)
        reward_batch = torch.FloatTensor(np.array(batch.reward))
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(np.array(batch.done))
        
        # Calculate current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.loss_history.append(loss.item())
            
        return loss.item()