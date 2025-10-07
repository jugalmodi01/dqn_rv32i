# RISC-V RL-based Instruction Generator

This project implements a Reinforcement Learning (RL) approach to generate RISC-V instructions for improving functional coverage of a RV32I processor. The system uses a Deep Q-Network (DQN) to learn which instructions will maximize coverage gains, resulting in non-redundant test cases that target specific coverage areas.

## Overview

Traditional constrained random instruction generation can be inefficient, often producing redundant test cases. This project introduces a more intelligent approach using RL to:

1. Analyze current coverage data
2. Learn which instruction types need more coverage
3. Generate optimal instructions to improve coverage
4. Avoid redundant instruction generation

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, tqdm (see requirements.txt)

## Project Structure

```
project/
│
├── requirements.txt          # Python dependencies
├── generate_instructions.py  # Main script for training and generating instructions
├── riscv_coverage_analyzer.py # Coverage data parser and analyzer
├── riscv_instruction_generator.py # RISC-V instruction encoder
├── dqn_model.py              # Deep Q-Network implementation
├── test_rl_generator.py      # Testing script for components
│
├── cov_data.csv              # Input coverage data
└── instruction.txt           # Output generated instructions
```

## Implementation Steps

### 1. Set up the environment

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Understand the coverage format

The system works with the coverage data format specified in `cov_data.csv`, which contains:
- Instruction type coverage (r_type, i_type, s_type, b_type, u_type, j_type)
- Coverage bins for specific instructions and their fields
- Coverage percentages for each bin

### 3. Test the components

Run the test script to verify that all components are working correctly:

```bash
python test_rl_generator.py
```

This will:
- Parse the coverage data from `cov_data.csv`
- Show the current coverage status
- Test generating instructions of various types
- Create a sample DQN agent

### 4. Train the DQN and generate instructions

Run the main script to train the DQN and generate optimized instructions:

```bash
python generate_instructions.py --coverage_file cov_data.csv --output_file instruction.txt --episodes 500
```

Parameters:
- `--coverage_file`: Path to the input coverage data
- `--output_file`: Path to save generated instructions
- `--episodes`: Number of training episodes
- `--instructions_per_episode`: Maximum instructions to generate per episode

### 5. Analyze the results

After running, the script will:
- Save generated instructions to `instruction.txt`
- Create a plot of training rewards in `dqn_training_rewards.png`
- Print a summary of the final coverage improvement

## How the DQN Model Works

The DQN model consists of:

1. **State Space**: Representation of current coverage for all bins (normalized to 0-1)

2. **Action Space**: All possible RISC-V instruction types that can be generated

3. **Reward Function**: Rewards for covering previously uncovered or low-coverage bins

4. **Network Architecture**: 
   - Input layer (state size)
   - Two hidden layers (128 neurons each)
   - Output layer (action size)
   
5. **Experience Replay**: Stores experiences (state, action, reward, next state) for batch learning

6. **Exploration Strategy**: Epsilon-greedy approach that gradually shifts from exploration to exploitation

7. **Redundancy Prevention**: Tracking mechanism to avoid generating duplicate instructions

## Key Components

### RISCVCoverageAnalyzer
- Parses `cov_data.csv` to extract coverage information
- Creates a structured representation of coverage data
- Provides methods to get uncovered bins and coverage state

### RISCVInstructionGenerator  
- Encodes different types of RISC-V instructions (R, I, S, B, U, J)
- Generates valid random instructions of specified types
- Formats instructions in hex for output

### DQNAgent
- Implements the Deep Q-Network algorithm
- Selects instructions using an epsilon-greedy strategy
- Learns from experience to maximize coverage improvement
- Avoids generating redundant instructions

## Future Improvements

1. **Interactive Simulation**: Connect directly to the simulator to get real coverage feedback
2. **More Advanced RL Algorithms**: Implement PPO or A3C for potentially better performance
3. **Coverage Cross-Bins**: Extend to handle coverage cross-bins
4. **Instruction Sequences**: Generate optimal sequences of instructions rather than individual ones

## References

- RISC-V Specification: https://riscv.org/technical/specifications/
- DQN Algorithm: https://www.nature.com/articles/nature14236