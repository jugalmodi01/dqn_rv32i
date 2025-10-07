import pandas as pd
import numpy as np

class RISCVCoverageAnalyzer:
    def __init__(self, coverage_file='cov_data.csv'):
        self.coverage_data = pd.read_csv(coverage_file)
        self.parse_coverage_data()
        
    def parse_coverage_data(self):
        """Parse the coverage data and create a structured representation"""
        self.coverage_metrics = {}
        self.instruction_types = ['r_type', 'i_type', 's_type', 'b_type', 'u_type', 'j_type']
        
        # Extract coverage data for each instruction type
        for instr_type in self.instruction_types:
            type_data = self.coverage_data[self.coverage_data['Name'].str.contains(f'riscv_coverage::{instr_type}_cg')]
            if not type_data.empty:
                self.coverage_metrics[instr_type] = {
                    'overall': float(type_data['Overall Average Grade'].values[0].strip('%')),
                    'bins': {}
                }
                
                # Get specific bins for this instruction type
                bins_data = self.coverage_data[self.coverage_data['Name'].str.contains(f'riscv_coverage::{instr_type}_cg')
                                              & self.coverage_data['Name'].str.contains('cp_')]
                
                for _, row in bins_data.iterrows():
                    bin_path = row['Name'].split('.')[-1]
                    self.coverage_metrics[instr_type]['bins'][bin_path] = {
                        'coverage': float(row['Overall Average Grade'].strip('%')),
                        'covered': int(row['Overall Covered'])
                    }
    
    def get_uncovered_bins(self):
        """Returns a list of uncovered bins"""
        uncovered = []
        for instr_type, data in self.coverage_metrics.items():
            for bin_name, bin_data in data['bins'].items():
                if bin_data['coverage'] < 100:
                    uncovered.append((instr_type, bin_name))
        return uncovered
    
    def get_coverage_state(self):
        """Returns a flattened state representation of all coverage bins"""
        state = []
        for instr_type in self.instruction_types:
            if instr_type in self.coverage_metrics:
                type_data = self.coverage_metrics[instr_type]
                for bin_name, bin_data in type_data['bins'].items():
                    state.append(bin_data['coverage'] / 100.0)  # Normalize to 0-1
            else:
                # If no data for this type, assume 0% coverage
                state.extend([0.0] * 5)  # Approximate number of bins
        return np.array(state, dtype=np.float32)
    
    def get_coverage_summary(self):
        """Returns a summary of the current coverage status"""
        summary = {}
        for instr_type in self.instruction_types:
            if instr_type in self.coverage_metrics:
                summary[instr_type] = self.coverage_metrics[instr_type]['overall']
            else:
                summary[instr_type] = 0.0
        return summary