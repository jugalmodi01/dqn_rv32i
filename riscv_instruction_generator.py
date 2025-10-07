import random

class RISCVInstructionGenerator:
    def __init__(self):
        # Define the instruction formats and fields
        self.r_type_ops = {
            'add': {'opcode': 0b0110011, 'funct3': 0b000, 'funct7': 0b0000000},
            'sub': {'opcode': 0b0110011, 'funct3': 0b000, 'funct7': 0b0100000},
            'sll': {'opcode': 0b0110011, 'funct3': 0b001, 'funct7': 0b0000000},
            'slt': {'opcode': 0b0110011, 'funct3': 0b010, 'funct7': 0b0000000},
            'sltu': {'opcode': 0b0110011, 'funct3': 0b011, 'funct7': 0b0000000},
            'xor': {'opcode': 0b0110011, 'funct3': 0b100, 'funct7': 0b0000000},
            'srl': {'opcode': 0b0110011, 'funct3': 0b101, 'funct7': 0b0000000},
            'sra': {'opcode': 0b0110011, 'funct3': 0b101, 'funct7': 0b0100000},
            'or': {'opcode': 0b0110011, 'funct3': 0b110, 'funct7': 0b0000000},
            'and': {'opcode': 0b0110011, 'funct3': 0b111, 'funct7': 0b0000000}
        }
        
        self.i_type_ops = {
            'addi': {'opcode': 0b0010011, 'funct3': 0b000},
            'slti': {'opcode': 0b0010011, 'funct3': 0b010},
            'sltiu': {'opcode': 0b0010011, 'funct3': 0b011},
            'xori': {'opcode': 0b0010011, 'funct3': 0b100},
            'ori': {'opcode': 0b0010011, 'funct3': 0b110},
            'andi': {'opcode': 0b0010011, 'funct3': 0b111},
            'slli': {'opcode': 0b0010011, 'funct3': 0b001, 'funct7': 0b0000000},
            'srli': {'opcode': 0b0010011, 'funct3': 0b101, 'funct7': 0b0000000},
            'srai': {'opcode': 0b0010011, 'funct3': 0b101, 'funct7': 0b0100000},
            # Load operations
            'lb': {'opcode': 0b0000011, 'funct3': 0b000},
            'lh': {'opcode': 0b0000011, 'funct3': 0b001},
            'lw': {'opcode': 0b0000011, 'funct3': 0b010},
            'lbu': {'opcode': 0b0000011, 'funct3': 0b100},
            'lhu': {'opcode': 0b0000011, 'funct3': 0b101},
            # Jump and link register
            'jalr': {'opcode': 0b1100111, 'funct3': 0b000}
        }
        
        self.s_type_ops = {
            'sb': {'opcode': 0b0100011, 'funct3': 0b000},
            'sh': {'opcode': 0b0100011, 'funct3': 0b001},
            'sw': {'opcode': 0b0100011, 'funct3': 0b010}
        }
        
        self.b_type_ops = {
            'beq': {'opcode': 0b1100011, 'funct3': 0b000},
            'bne': {'opcode': 0b1100011, 'funct3': 0b001},
            'blt': {'opcode': 0b1100011, 'funct3': 0b100},
            'bge': {'opcode': 0b1100011, 'funct3': 0b101},
            'bltu': {'opcode': 0b1100011, 'funct3': 0b110},
            'bgeu': {'opcode': 0b1100011, 'funct3': 0b111}
        }
        
        self.u_type_ops = {
            'lui': {'opcode': 0b0110111},
            'auipc': {'opcode': 0b0010111}
        }
        
        self.j_type_ops = {
            'jal': {'opcode': 0b1101111}
        }
        
        # Combine all operations for easier access
        self.all_ops = {}
        self.all_ops.update(self.r_type_ops)
        self.all_ops.update(self.i_type_ops)
        self.all_ops.update(self.s_type_ops)
        self.all_ops.update(self.b_type_ops)
        self.all_ops.update(self.u_type_ops)
        self.all_ops.update(self.j_type_ops)
        
        # Map instruction types
        self.instr_type_map = {}
        for op in self.r_type_ops:
            self.instr_type_map[op] = 'r_type'
        for op in self.i_type_ops:
            self.instr_type_map[op] = 'i_type'
        for op in self.s_type_ops:
            self.instr_type_map[op] = 's_type'
        for op in self.b_type_ops:
            self.instr_type_map[op] = 'b_type'
        for op in self.u_type_ops:
            self.instr_type_map[op] = 'u_type'
        for op in self.j_type_ops:
            self.instr_type_map[op] = 'j_type'
            
        # List of all operations for action space
        self.operation_list = list(self.all_ops.keys())
        
    def get_action_space_size(self):
        """Return the size of the action space (number of different instructions)"""
        return len(self.operation_list)
        
    def generate_instruction(self, operation):
        """Generate a single instruction based on the operation name"""
        if operation not in self.all_ops:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Get operation details
        op_details = self.all_ops[operation]
        instr_type = self.instr_type_map[operation]
        
        # Generate random register numbers
        rd = random.randint(0, 31)
        rs1 = random.randint(0, 31)
        rs2 = random.randint(0, 31)
        
        # Generate instruction based on type
        if instr_type == 'r_type':
            funct7 = op_details['funct7']
            funct3 = op_details['funct3']
            opcode = op_details['opcode']
            
            # Format: funct7[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
            instruction = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            
        elif instr_type == 'i_type':
            funct3 = op_details['funct3']
            opcode = op_details['opcode']
            
            # For shift instructions, imm[11:5] is funct7
            if operation in ['slli', 'srli', 'srai']:
                imm = random.randint(0, 31)  # Shift amount (0-31)
                funct7 = op_details.get('funct7', 0)
                imm = (funct7 << 5) | imm
            elif operation == 'jalr':
                # For jalr, set immediate to 4 (PC+4) to jump to the next instruction
                imm = 4 & 0xFFF  # 12-bit immediate
            else:
                # Normal immediate
                imm = random.randint(-2048, 2047) & 0xFFF  # 12-bit immediate
                
            # Format: imm[31:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
            instruction = (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            
        elif instr_type == 's_type':
            funct3 = op_details['funct3']
            opcode = op_details['opcode']
            
            # 12-bit immediate split between two fields
            imm = random.randint(0, 4095) & 0xFFF
            imm_11_5 = (imm >> 5) & 0x7F  # Upper 7 bits
            imm_4_0 = imm & 0x1F          # Lower 5 bits
            
            # Format: imm[11:5][31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[4:0][11:7] | opcode[6:0]
            instruction = (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode
            
        elif instr_type == 'b_type':
            funct3 = op_details['funct3']
            opcode = op_details['opcode']
            
            # For branch instructions, set immediate to 4 (PC+4) to branch to the next instruction
            # In B-type instructions, the immediate is in units of 2 bytes, so 4 bytes = offset of 2
            imm = 4  # 4 bytes (next instruction)
            
            # Extract bits for the B-type encoding
            # For imm=4, the encoded bits should be:
            imm_12 = 0
            imm_10_5 = 0
            imm_4_1 = 2  # 4 >> 1 = 2
            imm_11 = 0
            
            # Format: imm[12][31] | imm[10:5][30:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | 
            # imm[4:1][11:8] | imm[11][7] | opcode[6:0]
            instruction = (imm_12 << 31) | (imm_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
                         (funct3 << 12) | (imm_4_1 << 8) | (imm_11 << 7) | opcode
            
        elif instr_type == 'u_type':
            opcode = op_details['opcode']
            
            # 20-bit immediate (upper 20 bits of a 32-bit value)
            imm = random.randint(0, 0xFFFFF) & 0xFFFFF
            
            # Format: imm[31:12] | rd[11:7] | opcode[6:0]
            instruction = (imm << 12) | (rd << 7) | opcode
            
        elif instr_type == 'j_type':
            opcode = op_details['opcode']
            
            # For JAL, set immediate to 4 (PC+4) to jump to the next instruction
            # In J-type instructions, the immediate is in units of 2 bytes
            imm = 4  # 4 bytes (next instruction)
            
            # Extract bits for J-type encoding for imm=4
            imm_20 = 0
            imm_10_1 = 2  # 4 >> 1 = 2
            imm_11 = 0
            imm_19_12 = 0
            
            # Format: imm[20][31] | imm[10:1][30:21] | imm[11][20] | imm[19:12][19:12] | rd[11:7] | opcode[6:0]
            instruction = (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | \
                         (imm_19_12 << 12) | (rd << 7) | opcode
            
        return instruction, operation
    
    def format_instruction_hex(self, instruction):
        """Convert an instruction to a hex string format"""
        return f"0x{instruction:08x}"