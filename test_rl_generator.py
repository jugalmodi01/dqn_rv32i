from riscv_coverage_analyzer import RISCVCoverageAnalyzer
from riscv_instruction_generator import RISCVInstructionGenerator
from dqn_model import DQNAgent
import numpy as np

def main():
    # Initialize the coverage analyzer with the provided coverage data
    print("Initializing RISC-V coverage analyzer...")
    analyzer = RISCVCoverageAnalyzer('cov_data.csv')
    
    # Print the current coverage status
    coverage_summary = analyzer.get_coverage_summary()
    print("\nCurrent Coverage Summary:")
    for instr_type, coverage in coverage_summary.items():
        print(f"  {instr_type}: {coverage:.2f}%")
    
    # Print uncovered bins
    uncovered = analyzer.get_uncovered_bins()
    print(f"\nFound {len(uncovered)} uncovered bins")
    
    # Initialize the instruction generator
    print("\nInitializing RISC-V instruction generator...")
    generator = RISCVInstructionGenerator()
    
    # Test generating some instructions
    print("\nTesting instruction generation:")
    test_ops = ['add', 'beq', 'lw', 'sw', 'lui', 'jal']
    for op in test_ops:
        instruction, operation = generator.generate_instruction(op)
        print(f"  {op}: {generator.format_instruction_hex(instruction)}")
    
    # Create and test a simple DQN agent
    print("\nTesting DQN agent...")
    state_size = len(analyzer.get_coverage_state())
    action_size = generator.get_action_space_size()
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=128,
        learning_rate=0.001,
        memory_size=1000,
        batch_size=32
    )
    
    # Test action selection
    state = analyzer.get_coverage_state()
    action = agent.select_action(state, generator)
    op_name = generator.operation_list[action]
    print(f"  Selected action: {action}, Operation: {op_name}")
    
    print("\nSetup complete! Run generate_instructions.py to train the agent and generate instructions.")

if __name__ == "__main__":
    main()