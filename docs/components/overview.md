# Benchmark Components Overview

CFDLLMBench consists of three complementary evaluation tasks designed to comprehensively assess LLM capabilities in computational fluid dynamics.

## Design Philosophy

Our benchmark follows a hierarchical evaluation approach:

1. **Conceptual Understanding** (CFDQuery) - Foundation knowledge
2. **Code Generation** (CFDCodeBench) - Applied implementation  
3. **Tool Integration** (FoamBench) - End-to-end automation

## Component Breakdown

| Component | Type | Tasks | Evaluation Metric |
|-----------|------|-------|------------------|
| CFDQuery | Multiple Choice | 90 questions | Accuracy |
| CFDCodeBench | Code Generation | 24 problems | Execution + Similarity |
| FoamBench | Tool Use | 126 simulations | Success Rate |

## Evaluation Pipeline

```mermaid
graph LR
    A[Input Query] --> B[LLM Response]
    B --> C{Component Type}
    C -->|CFDQuery| D[Answer Matching]
    C -->|CFDCodeBench| E[Code Execution]
    C -->|FoamBench| F[Simulation Success]
    D --> G[Score Calculation]
    E --> G
    F --> G
```

## Learn More

- [CFDQuery Details](cfdquery.md)
- [CFDCodeBench Details](cfdcodebench.md) 
- [FoamBench Details](foambench.md)
