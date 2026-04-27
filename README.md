# cuda-assembler

Text-to-bytecode assembler for the agent instruction set. Supports labels, comments, data directives, confidence annotations, and the full 80-opcode ISA from cuda-instruction-set.

## Assembly Language

```text
  MOVI R0, 42        ; R0 = 42
  CONF R0, 0.95      ; R0 confidence = 0.95
  LABEL start:
  ADDI R1, R0, 1
  JNZ R1, start
  TELL R0, "hello"   ; A2A broadcast
  HALT
```

## Features

- **Labels**: `LABEL name:` / `JMP name`
- **Comments**: `; inline comments`
- **Data directives**: `DATA`, `STRING`, `CONF`
- **Registers**: R0–R7 general purpose
- **Confidence annotations**: attach confidence to any value
- **A2A instructions**: `TELL`, `ASK`, `BROADCAST` for inter-agent communication
- **Biological ops**: `INSTINCT`, `GENE_EXPR`, `ENZYME_BIND`, `ATP_GEN`

## Quick Start

```bash
git clone https://github.com/Lucineer/cuda-assembler.git
cd cuda-assembler
cargo test    # 15 tests
```

## Key Types

- **`Assembler`** — Two-pass assembler: collect labels, then emit bytecode
- **`Op`** — Internal opcode enum matching cuda-instruction-set
- **`Token`** — Lexed token with position tracking

## Dependencies

- `cuda-instruction-set` (peer dependency)

---

## Fleet Context

Part of the Lucineer/Cocapn fleet. See [fleet-onboarding](https://github.com/Lucineer/fleet-onboarding) for boarding protocol.

- **Vessel:** JetsonClaw1 (Jetson Orin Nano 8GB)
- **Domain:** Low-level systems, CUDA, edge computing
- **Comms:** Bottles via Forgemaster/Oracle1, Matrix #fleet-ops
