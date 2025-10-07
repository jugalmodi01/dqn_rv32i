import numpy as np
import torch
import argparse
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from riscv_coverage_analyzer import RISCVCoverageAnalyzer
from riscv_instruction_generator import RISCVInstructionGenerator
from dqn_model import DQNAgent

def simulate_coverage_increase(analyzer, generator, operation_name):
    """Simulate what the coverage would be if we add this instruction.
    This is a simple estimation since we can't actually run the simulation."""
    # Get instruction type from operation
    instr_type = generator.instr_type_map[operation_name]
    
    # Get current coverage summary
    current_coverage = analyzer.get_coverage_summary()
    
    # Increase coverage based on instruction type
    simulated_reward = 0
    
    # Higher reward for instruction types with lower coverage
    if instr_type in current_coverage:
        current_cov = current_coverage[instr_type]
        if current_cov < 100:
            # More reward for covering low-coverage areas
            simulated_reward = (100 - current_cov) / 10.0
    else:
        # Maximum reward for never-seen instruction types
        simulated_reward = 10.0
        
    # Add randomness to simulate uncertainty in coverage impact
    simulated_reward += np.random.normal(0, 1)
    
    return simulated_reward

def train_dqn_agent(analyzer, generator, num_episodes=500, max_instructions_per_episode=20,
                   save_path="instructions.txt"):
    """Train the DQN agent to generate instructions that maximize coverage"""
    # Get state size from analyzer
    state = analyzer.get_coverage_state()
    state_size = len(state)
    
    # Get action size from generator
    action_size = generator.get_action_space_size()
    
    # Create DQN agent
    agent = DQNAgent(state_size=state_size, 
                     action_size=action_size,
                     hidden_size=256,
                     learning_rate=0.001,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_end=0.01,
                     epsilon_decay=0.995,
                     memory_size=10000,
                     batch_size=64)
    
    # List to track rewards per episode
    rewards = []
    
    # List to store generated instructions
    generated_instructions = []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training DQN"):
        total_reward = 0
        state = analyzer.get_coverage_state()
        
        for step in range(max_instructions_per_episode):
            # Select action
            action = agent.select_action(state, generator)
            
            # Get the operation name for this action
            operation_name = generator.operation_list[action]
            
            # Generate an instruction for this operation
            instruction, _ = generator.generate_instruction(operation_name)
            instr_hex = generator.format_instruction_hex(instruction)
            
            # Simulate the coverage increase
            reward = simulate_coverage_increase(analyzer, generator, operation_name)
            
            # Store the instruction if it has a positive reward
            if reward > 0:
                generated_instructions.append((instr_hex, operation_name, reward))
                
            # Simplistic next state simulation (in reality, would run the simulation and get actual coverage)
            # Here we just slightly increase the coverage for the selected instruction type
            next_state = state.copy()
            instr_type = generator.instr_type_map[operation_name]
            
            # Estimate which bins might be affected by this instruction
            for i, cov in enumerate(next_state):
                if np.random.random() < 0.1:  # Small chance to affect any bin
                    next_state[i] = min(1.0, cov + 0.05)
                    
            # Learn from the experience
            agent.remember(state, action, next_state, reward, False)
            agent.learn()
            
            # Update state
            state = next_state
            
            # Update total reward
            total_reward += reward
        
        # Update epsilon for exploration
        agent.update_epsilon()
        rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {np.mean(rewards[-50:]):.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Sort instructions by their reward/coverage impact
    generated_instructions.sort(key=lambda x: x[2], reverse=True)
    
    # Save unique instructions to file - NO COMMENTS, just hex instructions
    seen_instructions = set()
    with open(save_path, 'w') as f:
        for instr, _, _ in generated_instructions:
            if instr not in seen_instructions:
                seen_instructions.add(instr)
                f.write(f"{instr[2:]}\n")  # Remove "0x" prefix and write only hex
                
    print(f"Generated {len(seen_instructions)} unique instructions and saved to {save_path}")
    
    # Plot the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('dqn_training_rewards.png')
    plt.close()
    
    return agent, rewards, generated_instructions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RISC-V Instruction Generator using DQN')
    parser.add_argument('--coverage_file', type=str, default='cov_data.csv',
                        help='Path to coverage data CSV file')
    parser.add_argument('--output_file', type=str, default='instruction.txt',
                        help='Path to output instruction file')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--instructions_per_episode', type=int, default=20,
                        help='Maximum instructions per episode')
    args = parser.parse_args()
    
    # Initialize analyzer and generator
    analyzer = RISCVCoverageAnalyzer(args.coverage_file)
    generator = RISCVInstructionGenerator()
    
    # Print current coverage summary
    coverage_summary = analyzer.get_coverage_summary()
    print("Current Coverage Summary:")
    for instr_type, cov in coverage_summary.items():
        print(f"  {instr_type}: {cov:.2f}%")
    
    # Train the agent and generate instructions
    agent, rewards, instructions = train_dqn_agent(
        analyzer, 
        generator, 
        num_episodes=args.episodes,
        max_instructions_per_episode=args.instructions_per_episode,
        save_path=args.output_file
    )
    
    print(f"Instruction generation complete! Instructions saved to {args.output_file}")