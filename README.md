# cuda-assembler

**Two-pass text-to-bytecode assembler for the FLUX instruction set.**

> Human-readable instructions in, machine-executable bytecode out.

## What It Does

Compiles text assembly into FLUX VM bytecode (used by flux-runtime-c):

```
# Text assembly
LOAD r1, 42       ; Load immediate
ADD  r2, r1, r3   ; Add registers
JZ   r2, end      ; Jump if zero
TELL "hello"      ; Output message
end: HALT          ; Stop
```

## Features

- **Two-pass assembly**: First pass collects labels, second pass emits bytes
- **60+ opcodes**: All FLUX opcodes including instinct and energy operations
- **Labels**: Named jump targets (`end:`, `loop:`)
- **Data directives**: Inline data embedding
- **Comments**: `;` line comments
- **Disassembler**: Bytecode back to text for debugging
- **Error reporting**: Line numbers and meaningful messages

## Ecosystem Integration

- [flux-runtime-c](https://github.com/Lucineer/flux-runtime-c) -- C VM that executes the bytecode
- [cuda-instruction-set](https://github.com/Lucineer/cuda-instruction-set) -- Rust opcode definitions
- [cuda-biology](https://github.com/Lucineer/cuda-biology) -- Generates instinct-based bytecode
- [cuda-genepool](https://github.com/Lucineer/cuda-genepool) -- Protein compilation to bytecode
- [cuda-forth](https://github.com/Lucineer/cuda-forth) -- Alternative: Forth-style compilation

## License

MIT OR Apache-2.0