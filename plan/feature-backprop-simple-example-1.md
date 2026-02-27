---
goal: Implement a simpler backpropagation example with a 1-to-1 node network in the LaTeX documentation.
version: 1.1
date_created: 2026-02-26
owner: GitHub Copilot
status: 'In progress'
tags: [feature, backprop, education]
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

The goal is to provide a simplified introduction to backpropagation within the existing educational document. This involves creating a section for a 1-input, 1-output neural network, detailing its mathematical steps (forward, loss, backward, update), and providing visual aids.

## 1. Requirements & Constraints

- **REQ-001**: Use a simple 1-to-1 node network.
- **REQ-002**: Fully explain the math (Forward pass, Loss function, Backward pass/Chain rule, Weight update).
- **REQ-003**: Generate a networkx-based graph for the simple network.
- **REQ-004**: Integrate the example into the existing LaTeX document before the advanced example.
- **CON-001**: Maintain consistency with existing notation (sigmoid activation, learning rate η).
- **CON-002**: Must activate `bmtk` conda environment before running Python scripts.

## 2. Implementation Steps

### Phase 1: Preparation & Visuals

| Task     | Description                                                                 | Completed | Date       |
| -------- | --------------------------------------------------------------------------- | --------- | ---------- |
| TASK-001 | Create `plot_simple_network.py` to generate the 1-to-1 network diagram. | ✅        | 2026-02-26 |
| TASK-002 | Run the script using `conda activate bmtk` to generate `simple_network.png`. | ✅        | 2026-02-26 |

### Phase 2: Documentation & Math

| Task     | Description                                                                 | Completed | Date       |
| -------- | --------------------------------------------------------------------------- | --------- | ---------- |
| TASK-003 | Update `main.tex` to include the "Simple 1-to-1 Example" section.         | ✅        | 2026-02-26 |
| TASK-004 | Create `solve_simple_network.py` to verify the manual calculations.      |           |            |
| TASK-005 | Verify all numerical values in `main.tex` against the script output.     |           |            |

### Phase 3: Finalization

| Task     | Description                                                                 | Completed | Date       |
| -------- | --------------------------------------------------------------------------- | --------- | ---------- |
| TASK-006 | Compile `main.tex` to PDF and ensure layout is correct.                   |           |            |

## 3. Alternatives

- **ALT-001**: Use a 2-1 network (2 inputs, 1 output). Rejected as it complicates the chain rule for the very first introduction.
- **ALT-002**: Use ReLU activation. Rejected to maintain consistency with the subsequent sigmoid-based example.

## 4. Dependencies

- **DEP-001**: Python libraries `networkx` and `matplotlib` (vailable in `bmtk` environment).
- **DEP-002**: LaTeX environment (`pdflatex`).

## 5. Files

- **FILE-001**: `/home/gjgpb9/Neural-Networks-Machine-Learning/write_ups_for_class/backprop/plot_simple_network.py` (New script)
- **FILE-002**: `/home/gjgpb9/Neural-Networks-Machine-Learning/write_ups_for_class/backprop/simple_network.png` (Generated image)
- **FILE-003**: `/home/gjgpb9/Neural-Networks-Machine-Learning/write_ups_for_class/backprop/solve_simple_network.py` (New verification script)
- **FILE-004**: `/home/gjgpb9/Neural-Networks-Machine-Learning/write_ups_for_class/backprop/main.tex` (Updated document)

## 6. Testing

- **TEST-001**: Run `solve_simple_network.py` in `bmtk` env to confirm output $ and gradient $\frac{\partial L}{\partial w}$.
- **TEST-002**: Compile LaTeX and verify image placement and formula rendering.

## 7. Risks & Assumptions

- **ASSUMPTION-001**: Conda is initialized and `bmtk` environment exists.
- **RISK-001**: Large images might push content to new pages in LaTeX, requiring layout adjustments.

## 8. Related Specifications / Further Reading

- Existing backprop example: [write_ups_for_class/backprop/main.tex](write_ups_for_class/backprop/main.tex)
