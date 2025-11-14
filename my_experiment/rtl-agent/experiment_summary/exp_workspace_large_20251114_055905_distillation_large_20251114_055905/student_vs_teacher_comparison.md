# Student vs Teacher Output Comparison

This document provides a side-by-side comparison of what the student model generated versus what the teacher model (reference solution) should generate.

---

## Problem: Fixed Priority Arbiter

**Task:** Design a priority arbiter that grants access to one of 8 requesters based on fixed priority (bit 0 = highest priority).

---

## Interface Comparison

### Required Interface (from Specification)

```verilog
module fixed_priority_arbiter(
    input clk,
    input reset,                    // Active-high synchronous reset
    input [7:0] req,
    input [7:0] priority_override,  // 8-bit vector
    output reg [7:0] grant,
    output reg valid,
    output reg [2:0] grant_index
);
```

### Student Model Output (WRONG)

```systemverilog
module fixed_priority_arbiter #(
    parameter int NUM_REQS = 4      // ❌ Should be 8, not parameterized
) (
    input  logic clk,
    input  logic rst_n,              // ❌ Should be "reset", active-high
    input  logic [NUM_REQS-1:0] req,
    input  logic priority_override,  // ❌ Should be [7:0] vector, not scalar
    output logic [NUM_REQS-1:0] grant,
    output logic grant_valid         // ❌ Should be "valid"
    // ❌ MISSING: grant_index output completely missing!
);
```

### Signal Name Mapping

| Specification | Student Generated | Status |
|---------------|-------------------|--------|
| `reset` | `rst_n` | ❌ Wrong name |
| `valid` | `grant_valid` | ❌ Wrong name |
| `grant_index [2:0]` | *missing* | ❌ Not implemented |
| `priority_override [7:0]` | `priority_override` (1-bit) | ❌ Wrong type |
| 8-bit width | `NUM_REQS=4` parameter | ❌ Wrong width |

**Result:** Test harness cannot connect to module due to signal name/type mismatches.

---

## Priority Logic Comparison

### Correct Logic (Teacher/Reference)

```verilog
// Priority: Bit 0 = HIGHEST, Bit 7 = LOWEST
always @(posedge clk or posedge reset) begin
    if (reset) begin
        grant <= 8'b00000000;
        valid <= 1'b0;
        grant_index <= 3'b000;
    end else begin
        if (priority_override != 8'b00000000) begin
            // Handle priority override (8-bit vector)
            grant <= priority_override;
            valid <= 1'b1;
            grant_index <= (priority_override[0] ? 3'd0 :
                           priority_override[1] ? 3'd1 :
                           priority_override[2] ? 3'd2 :
                           priority_override[3] ? 3'd3 :
                           priority_override[4] ? 3'd4 :
                           priority_override[5] ? 3'd5 :
                           priority_override[6] ? 3'd6 :
                           priority_override[7] ? 3'd7 : 3'd0);
        end
        // Fixed priority chain: bit 0 first (highest)
        else if (req[0]) begin
            grant <= 8'b00000001;
            grant_index <= 3'd0;
            valid <= 1'b1;
        end
        else if (req[1]) begin
            grant <= 8'b00000010;
            grant_index <= 3'd1;
            valid <= 1'b1;
        end
        else if (req[2]) begin
            grant <= 8'b00000100;
            grant_index <= 3'd2;
            valid <= 1'b1;
        end
        else if (req[3]) begin
            grant <= 8'b00001000;
            grant_index <= 3'd3;
            valid <= 1'b1;
        end
        else if (req[4]) begin
            grant <= 8'b00010000;
            grant_index <= 3'd4;
            valid <= 1'b1;
        end
        else if (req[5]) begin
            grant <= 8'b00100000;
            grant_index <= 3'd5;
            valid <= 1'b1;
        end
        else if (req[6]) begin
            grant <= 8'b01000000;
            grant_index <= 3'd6;
            valid <= 1'b1;
        end
        else if (req[7]) begin
            grant <= 8'b10000000;
            grant_index <= 3'd7;
            valid <= 1'b1;
        end
        else begin
            grant <= 8'b00000000;
            grant_index <= 3'd0;
            valid <= 1'b0;
        end
    end
end
```

### Student Logic (WRONG)

```systemverilog
// Priority: MSB first (BACKWARDS!)
always_ff @(posedge clk or negedge rst_n) begin  // ❌ Active-low async reset
    if (!rst_n) begin  // ❌ Wrong reset polarity
        grant <= '0;
        grant_valid <= '0;
        // ❌ Missing: grant_index
    end else begin
        if (priority_override) begin  // ❌ Treats as boolean, not vector
            // ❌ Wrong: does normal arbitration with MSB-first priority
            for (int i = NUM_REQS-1; i >= 0; i--) begin
                if (req[i]) begin
                    grant <= (1 << i);
                    grant_valid <= 1'b1;
                    break;
                end
            end
        end else begin
            // ❌ Scans from MSB to LSB (BACKWARDS!)
            for (int i = NUM_REQS-1; i >= 0; i--) begin
                if (req[i]) begin
                    grant <= (1 << i);
                    grant_valid <= 1'b1;
                    break;
                end
            end
        end
    end
end
```

---

## Test Case Comparison

### Test Case 1: Single Request

**Input:**
```
req = 0b00001000  (bit 3)
priority_override = 0b00000000
```

**Expected Output (Correct):**
```
grant = 0b00001000
grant_index = 3
valid = 1
```

**Student Output:**
```
Cannot compile - signal name mismatches
Test harness looks for "reset", "valid", "grant_index"
Student has "rst_n", "grant_valid", missing grant_index
RESULT: Compilation error ❌
```

---

### Test Case 2: Multiple Requests (Priority Test)

**Input:**
```
req = 0b00111000  (bits 3, 4, 5 requesting)
priority_override = 0b00000000
```

**Expected Output (Correct):**
```
grant = 0b00001000  // Bit 3 wins (lowest index = highest priority)
grant_index = 3
valid = 1
```

**Student Would Output (if it compiled):**
```
grant = 0b00100000  // ❌ Bit 5 wins (highest index)
grant_index = ???   // ❌ Missing this output
valid = 1 (as grant_valid)

Reason: Student scans from i=4 down to i=0
        Finds bit 5 first (because NUM_REQS=4 means i goes 3→0)

RESULT: Wrong priority order ❌
```

---

### Test Case 3: Priority Override

**Input:**
```
req = 0b00010010  (bits 1, 4 requesting)
priority_override = 0b00010000  (force grant to bit 4)
```

**Expected Output (Correct):**
```
grant = 0b00010000  // Override forces bit 4
grant_index = 4
valid = 1
```

**Student Would Output (if it compiled):**
```
Type error: priority_override is 1-bit, cannot compare to 8-bit vector
RESULT: Compilation error ❌

Even if types matched, student treats priority_override as a boolean flag,
not as a vector indicating which bit to grant
```

---

### Test Case 4: No Requests

**Input:**
```
req = 0b00000000
priority_override = 0b00000000
```

**Expected Output (Correct):**
```
grant = 0b00000000
grant_index = 0
valid = 0  // No grant issued
```

**Student Would Output (if it compiled):**
```
grant = 0b00000000
grant_index = ???  // Missing
grant_valid = 0

RESULT: Partially correct, but missing grant_index ❌
```

---

### Test Case 5: Highest vs Lowest Priority

**Input:**
```
req = 0b10000001  (bits 0 and 7 requesting)
priority_override = 0b00000000
```

**Expected Output (Correct):**
```
grant = 0b00000001  // Bit 0 wins (highest priority)
grant_index = 0
valid = 1
```

**Student Would Output (if it compiled):**
```
grant = 0b10000000  // ❌ Bit 7 wins (scans MSB-first)
grant_index = ???   // Missing
grant_valid = 1

RESULT: Wrong priority ❌
```

---

## Error Summary

### Category 1: Interface Errors (CRITICAL)

| Error | Impact | Fix Difficulty |
|-------|--------|----------------|
| Signal `reset` → `rst_n` | Test harness can't connect | Easy |
| Signal `valid` → `grant_valid` | Test harness can't connect | Easy |
| Missing `grant_index` output | Test fails immediately | Medium |
| `priority_override` is 1-bit not 8-bit | Type mismatch | Easy |
| Width is parameterized not fixed 8 | Wrong bus width | Easy |

**Why Critical:** Tests cannot even run if signals don't match.

### Category 2: Logic Errors (CRITICAL)

| Error | Impact | Fix Difficulty |
|-------|--------|----------------|
| MSB-first priority (backwards) | Wrong arbitration winner | Medium |
| priority_override treated as boolean | Feature doesn't work | Medium |
| Active-low async reset | Wrong reset behavior | Easy |

**Why Critical:** Even if interface matched, logic would fail all functional tests.

### Category 3: Style Differences (ACCEPTABLE)

| Difference | Impact | Fix Difficulty |
|------------|--------|----------------|
| SystemVerilog vs Verilog | None - both valid | N/A |
| `logic` vs `reg` | None - both valid | N/A |
| Modern `always_ff` vs `always` | None - both valid | N/A |

**Why Acceptable:** These are valid engineering choices, don't affect functionality.

---

## Visual: Priority Order Comparison

### Correct Priority (Bit 0 = Highest)

```
Request bits:  [7] [6] [5] [4] [3] [2] [1] [0]
Priority:      LOW ← ← ← ← ← ← ← ← HIGH

Arbitration scan order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
First match wins.

Example: req = 0b00111000 (bits 3,4,5 active)
         Scan: 0(no) → 1(no) → 2(no) → 3(YES!) → STOP
         Winner: bit 3 ✅
```

### Student's Priority (MSB = Highest - WRONG)

```
Request bits:  [7] [6] [5] [4] [3] [2] [1] [0]
Priority:      HIGH → → → → → → → → LOW

Arbitration scan order: 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0
First match wins.

Example: req = 0b00111000 (bits 3,4,5 active)
         Scan: 7(no) → 6(no) → 5(YES!) → STOP
         Winner: bit 5 ❌ WRONG!
```

---

## Code Quality Assessment

### What Student Did Well

1. ✅ **Clean SystemVerilog syntax** - No syntax errors
2. ✅ **Parameterized design** - Flexible width (good engineering practice)
3. ✅ **Modern coding style** - Used `logic`, `always_ff`, etc.
4. ✅ **Proper formatting** - Well-indented, readable
5. ✅ **Type safety** - Used `int` for loop variables

### What Student Did Wrong

1. ❌ **Didn't read specification carefully** - Wrong signal names
2. ❌ **Made assumptions** - Assumed "normal" conventions (MSB-first, rst_n)
3. ❌ **Incomplete implementation** - Missing grant_index output
4. ❌ **Misunderstood requirements** - priority_override as boolean vs vector
5. ❌ **Wrong priorities** - MSB-first instead of LSB-first

---

## Root Cause Analysis

### Why did the student make these mistakes?

#### Hypothesis 1: Training Data Bias
- RTL code in training data likely uses varied signal naming conventions
- Common to see `rst`, `rst_n`, `reset`, `areset`, etc.
- Common to see `valid`, `ready`, `enable`, `en`, etc.
- Model learned "these are all equivalent"
- Didn't learn "exact specification compliance is critical"

#### Hypothesis 2: Common Practice Defaults
- **MSB-first priority** is common in many arbiters (e.g., interrupt controllers)
- **Active-low resets** (`rst_n`) are ASIC design standards
- **Parameterized widths** are considered "good engineering"
- Model learned "best practices" but not "follow spec exactly"

#### Hypothesis 3: Incomplete Specification Reading
- The specification document is long (multiple sections)
- Model may have skimmed key details
- Focused on high-level concept ("build an arbiter") not exact requirements
- Didn't carefully parse the interface table

#### Hypothesis 4: No Iterative Refinement
- Generated code once
- No feedback loop to see test failures
- No opportunity to say "oh, I got the priority wrong, let me fix it"
- One-shot generation without iteration

---

## Key Takeaway

**The student model knows how to write RTL code, but doesn't know how to follow specifications exactly.**

This is a **specification compliance problem**, not a **code generation problem**.

**What the model needs to learn:**
1. Signal names must match specification exactly (not "similar" names)
2. Data types must match specification exactly (not "reasonable" alternatives)
3. All outputs must be implemented (not "optional" based on engineering judgment)
4. Behavior must match specification exactly (not "industry best practices")

**How to teach this:**
- More diverse training examples with strict specification compliance
- Reward structure that penalizes even small deviations
- Prompts that emphasize "EXACT compliance required"
- Iterative refinement with test feedback

---

## Appendix: Full Side-by-Side Code

### Teacher (Correct) - Full Implementation

```verilog
`timescale 1ns / 1ps
module fixed_priority_arbiter(
    input clk,
    input reset,
    input [7:0] req,
    input [7:0] priority_override,

    output reg [7:0] grant,
    output reg valid,
    output reg [2:0] grant_index
);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            grant <= 8'b00000000;
            valid <= 1'b0;
            grant_index <= 3'b000;
        end
        else begin
            if (priority_override != 8'b00000000) begin
                grant <= priority_override;
                valid <= 1'b1;
                grant_index <= (priority_override[0] ? 3'd0 :
                                priority_override[1] ? 3'd1 :
                                priority_override[2] ? 3'd2 :
                                priority_override[3] ? 3'd3 :
                                priority_override[4] ? 3'd4 :
                                priority_override[5] ? 3'd5 :
                                priority_override[6] ? 3'd6 :
                                priority_override[7] ? 3'd7 : 3'd0);
            end
            else if (req[0]) begin
                grant <= 8'b00000001;
                grant_index <= 3'd0;
                valid <= 1'b1;
            end
            else if (req[1]) begin
                grant <= 8'b00000010;
                grant_index <= 3'd1;
                valid <= 1'b1;
            end
            else if (req[2]) begin
                grant <= 8'b00000100;
                grant_index <= 3'd2;
                valid <= 1'b1;
            end
            else if (req[3]) begin
                grant <= 8'b00001000;
                grant_index <= 3'd3;
                valid <= 1'b1;
            end
            else if (req[4]) begin
                grant <= 8'b00010000;
                grant_index <= 3'd4;
                valid <= 1'b1;
            end
            else if (req[5]) begin
                grant <= 8'b00100000;
                grant_index <= 3'd5;
                valid <= 1'b1;
            end
            else if (req[6]) begin
                grant <= 8'b01000000;
                grant_index <= 3'd6;
                valid <= 1'b1;
            end
            else if (req[7]) begin
                grant <= 8'b10000000;
                grant_index <= 3'd7;
                valid <= 1'b1;
            end
            else begin
                grant <= 8'b00000000;
                grant_index <= 3'd0;
                valid <= 1'b0;
            end
        end
    end
endmodule
```

### Student (Wrong) - Reconstructed Implementation

```systemverilog
module fixed_priority_arbiter #(
    parameter int NUM_REQS = 4
) (
    input  logic clk,
    input  logic rst_n,
    input  logic [NUM_REQS-1:0] req,
    input  logic priority_override,
    output logic [NUM_REQS-1:0] grant,
    output logic grant_valid
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        grant <= '0;
        grant_valid <= '0;
    end else begin
        grant <= '0;
        grant_valid <= '0;

        if (priority_override) begin
            // MSB-first scan
            for (int i = NUM_REQS-1; i >= 0; i--) begin
                if (req[i]) begin
                    grant[i] <= 1'b1;
                    grant_valid <= 1'b1;
                    break;
                end
            end
        end else begin
            // MSB-first scan
            for (int i = NUM_REQS-1; i >= 0; i--) begin
                if (req[i]) begin
                    grant[i] <= 1'b1;
                    grant_valid <= 1'b1;
                    break;
                end
            end
        end
    end
end

endmodule
```

**Visual Diff:**
- ❌ 6 signal name/type mismatches
- ❌ 1 missing output (grant_index)
- ❌ Wrong priority order (MSB-first)
- ❌ Wrong reset polarity/timing
- ❌ Wrong priority_override handling

---

**Document Version:** 1.0
**Companion to:** experiment_summary/README.md
