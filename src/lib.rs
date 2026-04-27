/*!
# cuda-assembler

Text-to-bytecode assembler for agent instruction sets.

Supports: labels, comments, data directives, confidence annotations,
and the full 80-opcode agent instruction set from cuda-instruction-set.

Usage (pseudocode, not Rust):
```text
  MOVI R0, 42        ; R0 = 42
  CONF R0, 0.95      ; R0 confidence = 0.95
  LABEL start:
  ADDI R1, R0, 1
  JNZ R1, start
  TELL R0, "hello"   ; A2A broadcast
  HALT
```
*/

use std::collections::HashMap;
use std::fmt;

/// Opcode enum matching cuda-instruction-set
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    // Control flow
    Nop=0x00, Halt=0x01, Jmp=0x02, Jz=0x03, Jnz=0x04, Je=0x05, Jne=0x06,
    Call=0x07, Ret=0x08,
    // Arithmetic (confidence-preserving)
    CAdd=0x10, CSub=0x11, CMul=0x12, CDiv=0x13, CMod=0x14, CNeg=0x15,
    CInc=0x16, CDec=0x17, CMin=0x18, CMax=0x19, CAbs=0x1A,
    // Comparison
    Cmp=0x20, CLt=0x21, CLe=0x22, CEq=0x23, CGt=0x24, CGe=0x25,
    // Bitwise
    And=0x30, Or=0x31, Xor=0x32, Not=0x33, Shl=0x34, Shr=0x35,
    // Float
    FCvt=0x38, FNeg=0x39, FAdd=0x3A, FSub=0x3B, FMul=0x3C, FDiv=0x3D,
    // Stack
    Push=0x40, Pop=0x41, Dup=0x42, Swap=0x43,
    // Memory
    Load=0x48, Store=0x49, LoadF=0x4A, StoreF=0x4B, MAlloc=0x4C, MFree=0x4D,
    // Confidence
    Conf=0x50, Fuse=0x51, Drop=0x52, Trust=0x53, Gate=0x54,
    // Immediate
    MovI=0x58, AddI=0x59,
    // A2A
    Tell=0x60, Ask=0x61, Broadcast=0x62, Listen=0x63, Delegate=0x64,
    // Instinct (biological)
    InstinctAct=0x68, InstinctQ=0x69,
    // Gene (biological)
    GeneExpr=0x6A, EnzymeBind=0x6B, RnaTrans=0x6C, ProteinFold=0x6D,
    // Membrane (biological)
    MembraneChk=0x6E, Quarantine=0x6F,
    // Energy (biological)
    AtpGen=0x70, AtpConsume=0x71, AtpQ=0x72, AtpTransfer=0x73,
    ApoptosisChk=0x74, ApoptosisTrigger=0x75,
    CircadianSet=0x76, CircadianGet=0x77,
    // System
    SysCall=0x78, Debug=0x79, Yield=0x7A,
}

impl Op {
    fn from_name(name: &str) -> Option<Self> {
        let m = [
            ("NOP", Op::Nop), ("HALT", Op::Halt), ("JMP", Op::Jmp), ("JZ", Op::Jz),
            ("JNZ", Op::Jnz), ("JE", Op::Je), ("JNE", Op::Jne), ("CALL", Op::Call),
            ("RET", Op::Ret),
            ("CADD", Op::CAdd), ("CSUB", Op::CSub), ("CMUL", Op::CMul), ("CDIV", Op::CDiv),
            ("CMOD", Op::CMod), ("CNEG", Op::CNeg), ("CINC", Op::CInc), ("CDEC", Op::CDec),
            ("CMIN", Op::CMin), ("CMAX", Op::CMax), ("CABS", Op::CAbs),
            ("CMP", Op::Cmp), ("CLT", Op::CLt), ("CLE", Op::CLe), ("CEQ", Op::CEq),
            ("CGT", Op::CGt), ("CGE", Op::CGe),
            ("AND", Op::And), ("OR", Op::Or), ("XOR", Op::Xor), ("NOT", Op::Not),
            ("SHL", Op::Shl), ("SHR", Op::Shr),
            ("FCVT", Op::FCvt), ("FNEG", Op::FNeg), ("FADD", Op::FAdd), ("FSUB", Op::FSub),
            ("FMUL", Op::FMul), ("FDIV", Op::FDiv),
            ("PUSH", Op::Push), ("POP", Op::Pop), ("DUP", Op::Dup), ("SWAP", Op::Swap),
            ("LOAD", Op::Load), ("STORE", Op::Store), ("LOADF", Op::LoadF),
            ("STOREF", Op::StoreF), ("MALLOC", Op::MAlloc), ("MFREE", Op::MFree),
            ("CONF", Op::Conf), ("FUSE", Op::Fuse), ("DROP", Op::Drop), ("TRUST", Op::Trust),
            ("GATE", Op::Gate),
            ("MOVI", Op::MovI), ("ADDI", Op::AddI),
            ("TELL", Op::Tell), ("ASK", Op::Ask), ("BROADCAST", Op::Broadcast),
            ("LISTEN", Op::Listen), ("DELEGATE", Op::Delegate),
            ("INSTINCT_ACT", Op::InstinctAct), ("INSTINCT_Q", Op::InstinctQ),
            ("GENE_EXPR", Op::GeneExpr), ("ENZYME_BIND", Op::EnzymeBind),
            ("RNA_TRANS", Op::RnaTrans), ("PROTEIN_FOLD", Op::ProteinFold),
            ("MEMBRANE_CHK", Op::MembraneChk), ("QUARANTINE", Op::Quarantine),
            ("ATP_GEN", Op::AtpGen), ("ATP_CONSUME", Op::AtpConsume), ("ATP_Q", Op::AtpQ),
            ("ATP_TRANSFER", Op::AtpTransfer),
            ("APOPTOSIS_CHK", Op::ApoptosisChk), ("APOPTOSIS_TRIGGER", Op::ApoptosisTrigger),
            ("CIRCADIAN_SET", Op::CircadianSet), ("CIRCADIAN_GET", Op::CircadianGet),
            ("SYSCALL", Op::SysCall), ("DEBUG", Op::Debug), ("YIELD", Op::Yield),
        ];
        m.iter().find(|(n,_)| n.eq_ignore_ascii_case(name)).map(|(_,o)| *o)
    }

    fn format(self) -> &'static str {
        match self {
            Op::Nop => "NOP", Op::Halt => "HALT", Op::Jmp => "JMP", Op::Jz => "JZ",
            Op::Jnz => "JNZ", Op::Je => "JE", Op::Jne => "JNE", Op::Call => "CALL",
            Op::Ret => "RET", Op::CAdd => "CADD", Op::CSub => "CSUB", Op::CMul => "CMUL",
            Op::CDiv => "CDIV", Op::CMod => "CMOD", Op::CNeg => "CNEG", Op::CInc => "CINC",
            Op::CDec => "CDEC", Op::CMin => "CMIN", Op::CMax => "CMAX", Op::CAbs => "CABS",
            Op::Cmp => "CMP", Op::CLt => "CLT", Op::CLe => "CLE", Op::CEq => "CEQ",
            Op::CGt => "CGT", Op::CGe => "CGE", Op::And => "AND", Op::Or => "OR",
            Op::Xor => "XOR", Op::Not => "NOT", Op::Shl => "SHL", Op::Shr => "SHR",
            Op::FCvt => "FCVT", Op::FNeg => "FNEG", Op::FAdd => "FADD", Op::FSub => "FSUB",
            Op::FMul => "FMUL", Op::FDiv => "FDIV", Op::Push => "PUSH", Op::Pop => "POP",
            Op::Dup => "DUP", Op::Swap => "SWAP", Op::Load => "LOAD", Op::Store => "STORE",
            Op::LoadF => "LOADF", Op::StoreF => "STOREF", Op::MAlloc => "MALLOC",
            Op::MFree => "MFREE", Op::Conf => "CONF", Op::Fuse => "FUSE", Op::Drop => "DROP",
            Op::Trust => "TRUST", Op::Gate => "GATE", Op::MovI => "MOVI", Op::AddI => "ADDI",
            Op::Tell => "TELL", Op::Ask => "ASK", Op::Broadcast => "BROADCAST",
            Op::Listen => "LISTEN", Op::Delegate => "DELEGATE",
            Op::InstinctAct => "INSTINCT_ACT", Op::InstinctQ => "INSTINCT_Q",
            Op::GeneExpr => "GENE_EXPR", Op::EnzymeBind => "ENZYME_BIND",
            Op::RnaTrans => "RNA_TRANS", Op::ProteinFold => "PROTEIN_FOLD",
            Op::MembraneChk => "MEMBRANE_CHK", Op::Quarantine => "QUARANTINE",
            Op::AtpGen => "ATP_GEN", Op::AtpConsume => "ATP_CONSUME", Op::AtpQ => "ATP_Q",
            Op::AtpTransfer => "ATP_TRANSFER", Op::ApoptosisChk => "APOPTOSIS_CHK",
            Op::ApoptosisTrigger => "APOPTOSIS_TRIGGER", Op::CircadianSet => "CIRCADIAN_SET",
            Op::CircadianGet => "CIRCADIAN_GET", Op::SysCall => "SYSCALL",
            Op::Debug => "DEBUG", Op::Yield => "YIELD",
        }
    }
}

/// Register name to index
fn parse_reg(s: &str) -> Option<u8> {
    let s = s.trim().trim_end_matches(',');
    if s.starts_with('R') || s.starts_with('r') {
        s[1..].parse::<u8>().ok().filter(|&r| r <= 15)
    } else {
        None
    }
}

/// Assembler error
#[derive(Debug)]
pub enum AsmError {
    UnknownOpcode(String),
    BadRegister(String),
    BadImmediate(String),
    UndefinedLabel(String),
    DuplicateLabel(String),
    ParseError(String),
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmError::UnknownOpcode(s) => write!(f, "unknown opcode: {}", s),
            AsmError::BadRegister(s) => write!(f, "bad register: {}", s),
            AsmError::BadImmediate(s) => write!(f, "bad immediate: {}", s),
            AsmError::UndefinedLabel(s) => write!(f, "undefined label: {}", s),
            AsmError::DuplicateLabel(s) => write!(f, "duplicate label: {}", s),
            AsmError::ParseError(s) => write!(f, "parse error: {}", s),
        }
    }
}

/// Two-pass assembler
pub struct Assembler {
    labels: HashMap<String, usize>,
    data: Vec<u8>,
}

impl Assembler {
    pub fn new() -> Self { Assembler { labels: HashMap::new(), data: vec![] } }

    /// Assemble source text into bytecode
    pub fn assemble(&mut self, source: &str) -> Result<Vec<u8>, Vec<AsmError>> {
        let mut errors = vec![];
        let lines: Vec<&str> = source.lines().collect();

        // Pass 1: collect labels and estimate sizes
        let mut pc = 0usize;
        for line in &lines {
            let line = line.split(';').next().unwrap_or("").trim();
            if line.is_empty() { continue; }
            if line.to_uppercase().starts_with("LABEL ") || line.ends_with(':') {
                // Handle both "LABEL foo:" and "foo:" formats
                let upper = line.to_uppercase();
                let name = if upper.starts_with("LABEL ") {
                    line[6..].trim().trim_end_matches(':').trim()
                } else {
                    line.trim_end_matches(':').trim()
                };
                if self.labels.contains_key(name) {
                    errors.push(AsmError::DuplicateLabel(name.to_string()));
                }
                self.labels.insert(name.to_string(), pc);
                continue;
            }
            if line.to_uppercase().starts_with("DATA ") {
                // Parse data bytes
                let bytes_str = &line[5..];
                let count = bytes_str.split(',').filter(|s| !s.trim().is_empty()).count();
                pc += count;
                continue;
            }
            let size = self.instr_size(line);
            if size == 0 && !line.is_empty() {
                if errors.len() < 10 { errors.push(AsmError::ParseError(format!("unknown: {}", line))); }
            }
            pc += size;
        }
        if !errors.is_empty() { return Err(errors); }

        // Pass 2: emit bytecode
        self.data = vec![];
        for line in &lines {
            let line = line.split(';').next().unwrap_or("").trim();
            if line.is_empty() { continue; }
            if line.to_uppercase().starts_with("LABEL ") || line.ends_with(':') { continue; }
            if line.to_uppercase().starts_with("DATA ") {
                self.emit_data(&line[5..]);
                continue;
            }
            self.emit_instruction(line, &mut errors);
        }
        if !errors.is_empty() { return Err(errors); }
        Ok(self.data.clone())
    }

    fn instr_size(&self, line: &str) -> usize {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { return 0; }
        let op_name = parts[0].to_uppercase();
        let op = match Op::from_name(&op_name) {
            Some(o) => o,
            None => return 0,
        };
        match op {
            // No operands: 1 byte
            Op::Nop | Op::Halt | Op::Ret | Op::Yield | Op::Swap => 1,
            // Single register: 2 bytes (op + rd)
            Op::CInc | Op::CDec | Op::CNeg | Op::CAbs | Op::Not |
            Op::Push | Op::Pop | Op::Dup |
            Op::InstinctQ | Op::AtpQ | Op::CircadianGet |
            Op::ApoptosisChk | Op::ApoptosisTrigger | Op::Debug => 2,
            // Two registers: 3 bytes (op + rd + rs1)
            Op::CAdd | Op::CSub | Op::CMul | Op::CDiv | Op::CMod |
            Op::CMin | Op::CMax | Op::Cmp | Op::CLt | Op::CLe | Op::CEq | Op::CGt | Op::CGe |
            Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr |
            Op::FCvt | Op::FNeg | Op::FAdd | Op::FSub | Op::FMul | Op::FDiv |
            Op::Load | Op::Store | Op::LoadF | Op::StoreF |
            Op::Conf | Op::Fuse | Op::Trust | Op::Gate |
            Op::GeneExpr | Op::EnzymeBind | Op::RnaTrans | Op::ProteinFold |
            Op::MembraneChk | Op::Quarantine |
            Op::AtpGen | Op::AtpConsume | Op::AtpTransfer |
            Op::CircadianSet | Op::InstinctAct |
            Op::Tell | Op::Ask | Op::Listen | Op::Delegate => 3,
            // Register + immediate: 4 bytes (op + rd + imm16)
            Op::MovI | Op::AddI | Op::MAlloc | Op::Drop | Op::Jz | Op::Jnz |
            Op::Je | Op::Jne => 4,
            // Jump: 4 bytes (op + flags + offset16)
            Op::Jmp | Op::Call => 4,
            // Broadcast: 5+ bytes (op + rd + len16 + bytes)
            Op::Broadcast => 5,
            // System: 3 bytes
            Op::SysCall => 3,
            Op::MFree => 2,
        }
    }

    fn emit_data(&mut self, s: &str) {
        for part in s.split(',') {
            let trimmed = part.trim();
            if let Ok(b) = trimmed.parse::<u8>() { self.data.push(b); }
        }
    }

    fn emit_instruction(&mut self, line: &str, errors: &mut Vec<AsmError>) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { return; }
        let op = match Op::from_name(parts[0]) {
            Some(o) => o,
            None => { errors.push(AsmError::UnknownOpcode(parts[0].to_string())); return; }
        };

        let get_reg = |idx: usize, errs: &mut Vec<AsmError>| -> u8 {
            if idx >= parts.len() { errs.push(AsmError::BadRegister("missing".into())); return 0; }
            parse_reg(parts[idx]).unwrap_or_else(|| { errs.push(AsmError::BadRegister(parts[idx].to_string())); 0 })
        };

        let _get_imm = |idx: usize, errs: &mut Vec<AsmError>| -> i16 {
            if idx >= parts.len() { errs.push(AsmError::BadImmediate("missing".into())); return 0; }
            // Try as label first
            if let Some(_addr) = self.labels.get(parts[idx]) {
                // Will be patched later — for now store 0
                return 0;
            }
            // Parse as number
            let s = parts[idx].trim_end_matches(',');
            s.parse::<i16>().unwrap_or_else(|_| {
                // Try hex
                if s.starts_with("0x") { i16::from_str_radix(&s[2..], 16).unwrap_or(0) }
                else { 0 }
            })
        };

        let _get_imm_from_label = |idx: usize, _errs: &mut Vec<AsmError>, labels: &HashMap<String,usize>, data_len: usize| -> i16 {
            if idx >= parts.len() { return 0; }
            let s = parts[idx].trim_end_matches(',');
            if let Some(&addr) = labels.get(s) {
                // offset from current PC (after this instruction)
                let instr_end = data_len + 4; // jumps are 4 bytes
                let offset = addr as isize - instr_end as isize;
                return offset as i16;
            }
            s.parse::<i16>().unwrap_or_else(|_| {
                if s.starts_with("0x") { i16::from_str_radix(&s[2..], 16).unwrap_or(0) }
                else { 0 }
            })
        };
        let labels_clone = self.labels.clone();

        self.data.push(op as u8);

        match op {
            Op::Nop | Op::Halt | Op::Ret | Op::Yield | Op::Swap => {}
            Op::CInc | Op::CDec | Op::CNeg | Op::CAbs | Op::Not |
            Op::Push | Op::Pop | Op::Dup | Op::MFree |
            Op::InstinctQ | Op::AtpQ | Op::CircadianGet |
            Op::ApoptosisChk | Op::ApoptosisTrigger | Op::Debug => {
                self.data.push(get_reg(1, errors));
            }
            Op::CAdd | Op::CSub | Op::CMul | Op::CDiv | Op::CMod |
            Op::CMin | Op::CMax | Op::Cmp | Op::CLt | Op::CLe | Op::CEq | Op::CGt | Op::CGe |
            Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr |
            Op::FCvt | Op::FNeg | Op::FAdd | Op::FSub | Op::FMul | Op::FDiv |
            Op::Load | Op::Store | Op::LoadF | Op::StoreF |
            Op::Conf | Op::Fuse | Op::Trust | Op::Gate |
            Op::GeneExpr | Op::EnzymeBind | Op::RnaTrans | Op::ProteinFold |
            Op::MembraneChk | Op::Quarantine |
            Op::AtpGen | Op::AtpConsume | Op::AtpTransfer |
            Op::CircadianSet | Op::InstinctAct |
            Op::Tell | Op::Ask | Op::Listen | Op::Delegate => {
                self.data.push(get_reg(1, errors));
                self.data.push(get_reg(2, errors));
            }
            Op::MovI | Op::AddI | Op::MAlloc | Op::Drop => {
                self.data.push(get_reg(1, errors));
                let imm = get_imm_from_label(2, errors, &labels_clone, self.data.len());
                self.data.push((imm & 0xFF) as u8);
                self.data.push(((imm >> 8) & 0xFF) as u8);
            }
            Op::Jz | Op::Jnz | Op::Je | Op::Jne => {
                self.data.push(get_reg(1, errors));
                let imm = get_imm_from_label(2, errors, &labels_clone, self.data.len());
                self.data.push((imm & 0xFF) as u8);
                self.data.push(((imm >> 8) & 0xFF) as u8);
            }
            Op::Jmp | Op::Call => {
                self.data.push(0); // flags/unused
                let imm = get_imm_from_label(1, errors, &labels_clone, self.data.len());
                self.data.push((imm & 0xFF) as u8);
                self.data.push(((imm >> 8) & 0xFF) as u8);
            }
            Op::Broadcast => {
                self.data.push(get_reg(1, errors));
                // For simplicity, emit empty payload
                self.data.push(0); self.data.push(0); // len
            }
            Op::SysCall => {
                let n = get_imm_from_label(1, errors, &labels_clone, self.data.len());
                self.data.push(n as u8);
                self.data.push(0);
            }
        }
    }

    /// Disassemble bytecode back to text
    pub fn disassemble(&self, bytecode: &[u8]) -> String {
        let mut out = String::new();
        let mut pc = 0;
        while pc < bytecode.len() {
            let op_byte = bytecode[pc];
            // Find matching op
            let op = Self::op_from_byte(op_byte);
            let name = op.map(|o| o.format()).unwrap_or("???");
            out.push_str(&format!("{:04x}: {}\n", pc, name));
            pc += match op {
                None => 1,
                Some(o) => self.op_size(o),
            };
        }
        out
    }

    fn op_from_byte(b: u8) -> Option<Op> {
        // Match by discriminant
        match b {
            0x00 => Some(Op::Nop), 0x01 => Some(Op::Halt), 0x02 => Some(Op::Jmp),
            0x03 => Some(Op::Jz), 0x04 => Some(Op::Jnz), 0x05 => Some(Op::Je),
            0x06 => Some(Op::Jne), 0x07 => Some(Op::Call), 0x08 => Some(Op::Ret),
            0x10 => Some(Op::CAdd), 0x11 => Some(Op::CSub), 0x12 => Some(Op::CMul),
            0x13 => Some(Op::CDiv), 0x14 => Some(Op::CMod), 0x15 => Some(Op::CNeg),
            0x16 => Some(Op::CInc), 0x17 => Some(Op::CDec), 0x18 => Some(Op::CMin),
            0x19 => Some(Op::CMax), 0x1A => Some(Op::CAbs),
            0x20 => Some(Op::Cmp), 0x21 => Some(Op::CLt), 0x22 => Some(Op::CLe),
            0x23 => Some(Op::CEq), 0x24 => Some(Op::CGt), 0x25 => Some(Op::CGe),
            0x30 => Some(Op::And), 0x31 => Some(Op::Or), 0x32 => Some(Op::Xor),
            0x33 => Some(Op::Not), 0x34 => Some(Op::Shl), 0x35 => Some(Op::Shr),
            0x38 => Some(Op::FCvt), 0x39 => Some(Op::FNeg), 0x3A => Some(Op::FAdd),
            0x3B => Some(Op::FSub), 0x3C => Some(Op::FMul), 0x3D => Some(Op::FDiv),
            0x40 => Some(Op::Push), 0x41 => Some(Op::Pop), 0x42 => Some(Op::Dup),
            0x43 => Some(Op::Swap),
            0x48 => Some(Op::Load), 0x49 => Some(Op::Store), 0x4A => Some(Op::LoadF),
            0x4B => Some(Op::StoreF), 0x4C => Some(Op::MAlloc), 0x4D => Some(Op::MFree),
            0x50 => Some(Op::Conf), 0x51 => Some(Op::Fuse), 0x52 => Some(Op::Drop),
            0x53 => Some(Op::Trust), 0x54 => Some(Op::Gate),
            0x58 => Some(Op::MovI), 0x59 => Some(Op::AddI),
            0x60 => Some(Op::Tell), 0x61 => Some(Op::Ask), 0x62 => Some(Op::Broadcast),
            0x63 => Some(Op::Listen), 0x64 => Some(Op::Delegate),
            0x68 => Some(Op::InstinctAct), 0x69 => Some(Op::InstinctQ),
            0x6A => Some(Op::GeneExpr), 0x6B => Some(Op::EnzymeBind),
            0x6C => Some(Op::RnaTrans), 0x6D => Some(Op::ProteinFold),
            0x6E => Some(Op::MembraneChk), 0x6F => Some(Op::Quarantine),
            0x70 => Some(Op::AtpGen), 0x71 => Some(Op::AtpConsume),
            0x72 => Some(Op::AtpQ), 0x73 => Some(Op::AtpTransfer),
            0x74 => Some(Op::ApoptosisChk), 0x75 => Some(Op::ApoptosisTrigger),
            0x76 => Some(Op::CircadianSet), 0x77 => Some(Op::CircadianGet),
            0x78 => Some(Op::SysCall), 0x79 => Some(Op::Debug), 0x7A => Some(Op::Yield),
            _ => None,
        }
    }

    fn op_size(&self, op: Op) -> usize {
        // Simplified — just return 1 for unknown
        let dummy = format!("{} R0", op.format());
        self.instr_size(&dummy).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_program() {
        let mut asm = Assembler::new();
        let src = "MOVI R0, 42\nCINC R0\nHALT";
        let bc = asm.assemble(src).unwrap();
        assert!(!bc.is_empty());
        assert_eq!(bc[0], Op::MovI as u8);
        assert_eq!(bc[4], Op::CInc as u8);
        assert_eq!(bc[6], Op::Halt as u8);
    }

    #[test]
    fn test_labels() {
        let mut asm = Assembler::new();
        let src = "LABEL start:\nMOVI R0, 10\nLABEL end:\nHALT";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(asm.labels["start"], 0);
        assert_eq!(asm.labels["end"], 4); // MOVI is 4 bytes
        assert_eq!(bc[4], Op::Halt as u8);
    }

    #[test]
    fn test_jump_to_label() {
        let mut asm = Assembler::new();
        let src = "LABEL loop:\nCDEC R0\nJNZ R0, loop\nHALT";
        let bc = asm.assemble(src).unwrap();
        assert!(bc.len() > 0);
    }

    #[test]
    fn test_comments() {
        let mut asm = Assembler::new();
        let src = "; this is a comment\nNOP ; inline comment\nHALT";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], Op::Nop as u8);
        assert_eq!(bc[1], Op::Halt as u8);
        assert_eq!(bc.len(), 2);
    }

    #[test]
    fn test_data_directive() {
        let mut asm = Assembler::new();
        let src = "DATA 1, 2, 3, 42\nHALT";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(&bc[0..4], &[1, 2, 3, 42]);
        assert_eq!(bc[4], Op::Halt as u8);
    }

    #[test]
    fn test_confidence_ops() {
        let mut asm = Assembler::new();
        let src = "CONF R0, R1\nFUSE R0, R1\nTRUST R0, R1";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], Op::Conf as u8);
        assert_eq!(bc[3], Op::Fuse as u8);
        assert_eq!(bc[6], Op::Trust as u8);
    }

    #[test]
    fn test_biological_ops() {
        let mut asm = Assembler::new();
        let src = "INSTINCT_ACT R0, R1\nGENE_EXPR R0, R1\nATP_GEN R0, R1\nMEMBRANE_CHK R0, R1";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], 0x68); // INSTINCT_ACT
        assert_eq!(bc[3], 0x6A); // GENE_EXPR
        assert_eq!(bc[6], 0x70); // ATP_GEN
        assert_eq!(bc[9], 0x6E); // MEMBRANE_CHK
    }

    #[test]
    fn test_a2a_ops() {
        let mut asm = Assembler::new();
        let src = "TELL R0, R1\nASK R0, R1\nBROADCAST R0";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], 0x60);
        assert_eq!(bc[3], 0x61);
        assert_eq!(bc[6], 0x62);
    }

    #[test]
    fn test_arithmetic() {
        let mut asm = Assembler::new();
        let src = "MOVI R0, 10\nMOVI R1, 3\nCADD R0, R1\nCDIV R0, R1";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], Op::MovI as u8);
        assert_eq!(bc[4], Op::MovI as u8);
        assert_eq!(bc[8], Op::CAdd as u8);
        assert_eq!(bc[11], Op::CDiv as u8);
    }

    #[test]
    fn test_case_insensitive() {
        let mut asm = Assembler::new();
        let bc1 = asm.assemble("NOP\nhalt").unwrap();
        let mut asm2 = Assembler::new();
        let bc2 = asm2.assemble("nop\nHALT").unwrap();
        assert_eq!(bc1, bc2);
    }

    #[test]
    fn test_unknown_opcode() {
        let mut asm = Assembler::new();
        let result = asm.assemble("FOOBAR R0, R1");
        assert!(result.is_err());
    }

    #[test]
    fn test_disassemble() {
        let mut asm = Assembler::new();
        let bc = asm.assemble("NOP\nMOVI R0, 42\nHALT").unwrap();
        let out = asm.disassemble(&bc);
        assert!(out.contains("NOP"));
        assert!(out.contains("MOVI"));
        assert!(out.contains("HALT"));
    }

    #[test]
    fn test_stack_ops() {
        let mut asm = Assembler::new();
        let src = "PUSH R0\nPUSH R1\nSWAP\nPOP R0";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], 0x40); // PUSH
        assert_eq!(bc[2], 0x40); // PUSH
        assert_eq!(bc[4], 0x43); // SWAP
        assert_eq!(bc[5], 0x41); // POP
    }

    #[test]
    fn test_memory_ops() {
        let mut asm = Assembler::new();
        let src = "MOVI R0, 0\nMOVI R1, 42\nSTORE R1, R0";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[8], Op::Store as u8);
    }

    #[test]
    fn test_call_ret() {
        let mut asm = Assembler::new();
        let src = "LABEL func:\nRET\nCALL func\nHALT";
        let bc = asm.assemble(src).unwrap();
        assert_eq!(bc[0], Op::Ret as u8);
        assert_eq!(bc[1], Op::Call as u8);
    }
}
