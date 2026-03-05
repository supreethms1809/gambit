# CDEA Framework Block Diagrams (Abstract)

This version is intentionally conceptual and paper-facing. It avoids implementation or code references.

## 1) Unified CDEA Framework (Abstract View)

```mermaid
flowchart TB
  IN["Inputs<br/>Model predictions + evidence signals + intervention space"] --> CORE

  subgraph CORE["CDEA Core (Unifying Layer)"]
    C1["Hypothesis framing<br/>Which alternatives are being explained?"]
    C2["Evidence structuring<br/>Map evidence into comparable units"]
    C3["Optional hypothesis interaction<br/>Model dependencies among hypotheses"]
    C4["Allocation game solver<br/>Optimize interpretable evidence partitions"]
    C5["Intervention-grounded evaluation<br/>Faithfulness, contrast, stability, parsimony"]
    C1 --> C2 --> C3 --> C4 --> C5
  end

  CORE --> I1
  CORE --> I2
  CORE -.-> IX1["..."]
  CORE -.-> IX2["Future instantiations"]
  CORE -.-> IX3["..."]

  subgraph I1["Instantiation I: Contrastive Class-Distribution Game"]
    I1A["Agents<br/>Top competing hypotheses (class players)"]
    I1G["Game form<br/>Shared-cooperative and unique-competitive allocation"]
    I1O["Outputs<br/>Shared evidence, unique evidence per hypothesis,<br/>pairwise why-k-rather-than-l, probability-split explanation"]
    I1A --> I1G --> I1O
  end

  subgraph I2["Instantiation II: Shift-Aware Robust-Shortcut Game"]
    I2A["Agents<br/>Robust evidence agent and shortcut evidence agent"]
    I2G["Game form<br/>Cross-environment allocation under distribution shift"]
    I2O["Outputs<br/>Robust evidence mask, shortcut evidence mask,<br/>stability and shortcut-gap diagnostics"]
    I2A --> I2G --> I2O
  end
```

## 2) Instantiation Semantics (Agent-Game-Output)

```mermaid
flowchart LR
  C["CDEA Core"] --> G1["Game 1<br/>Contrastive shared-unique"]
  C --> G2["Game 2<br/>Robust-shortcut under shift"]

  G1 --> A1["Agents<br/>Hypothesis players"]
  G1 --> O1["Outcome<br/>Contrastive evidence decomposition"]

  G2 --> A2["Agents<br/>Robust vs shortcut players"]
  G2 --> O2["Outcome<br/>Shift-aware robust/shortcut decomposition"]

  C -.-> GX["... additional games ..."]
```

## 3) Caption Starter (Abstract)

CDEA is a unifying explanation framework that turns raw class-conditioned evidence into game-theoretic evidence allocations. A shared core defines hypothesis framing, evidence structuring, optional interaction, allocation optimization, and intervention-grounded evaluation; distinct instantiations define agent sets and game objectives while producing structured, interpretable outputs.
