from pathlib import Path
import json
import pandas as pd
from datetime import datetime

ROOT = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")

RUNS = {
    "run02": ROOT / "outputs/run_02_q1_validation",
    "run03": ROOT / "outputs/run_03_q1_improvements_fixed",
    "run05": ROOT / "outputs/run_05_q1_psd_biological_validation",
    "run06": ROOT / "outputs/run_06_q1_levelup_biological_embedding_analysis",
    "run07": ROOT / "outputs/run_07_seedv_session_drift_subject_adaptation",
}

OUT = ROOT / "outputs/run_09_nvidia_agentic_research_platform"
TAB = OUT / "tables"
REP = OUT / "report"
LOG = OUT / "logs"

for p in [OUT, TAB, REP, LOG]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def safe_read_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
    return None


def find_files(folder, suffixes=(".csv", ".png", ".md", ".txt", ".json")):
    if not folder.exists():
        return []
    return [str(p) for p in folder.rglob("*") if p.suffix.lower() in suffixes]


# ============================================================
# Agent 1: Scientist Agent
# ============================================================

def scientist_agent():
    """
    Role:
    Suggests scientifically meaningful next experiments.
    Does not fabricate results.
    """
    suggestions = [
        {
            "priority": 1,
            "experiment": "AEP cross-dataset validation",
            "reason": "Tests whether cognitive/sensory-state EEG identity drift generalizes beyond SEED-V.",
            "expected_output": "External validation EER/AUC, drift-vs-EER, PSD drift, ROC/DET."
        },
        {
            "priority": 2,
            "experiment": "Statistical significance testing",
            "reason": "Strengthens Q1 journal validity using paired tests, confidence intervals, and FDR correction.",
            "expected_output": "p-values, effect sizes, corrected significance table."
        },
        {
            "priority": 3,
            "experiment": "Online subject adaptation",
            "reason": "Extends Run07 from offline adaptation to streaming biometric updating.",
            "expected_output": "time-to-stabilization, EER before/after adaptation, adaptation curves."
        }
    ]
    return suggestions


# ============================================================
# Agent 2: Critic Agent
# ============================================================

def critic_agent():
    """
    Role:
    Checks whether each run has necessary scientific evidence.
    """
    checks = []

    required_runs = {
        "run02": ["tables", "figures"],
        "run03": ["tables", "figures"],
        "run05": ["tables", "figures", "report"],
        "run06": ["tables", "figures", "report"],
        "run07": ["tables", "figures", "report"],
    }

    for run_name, folders in required_runs.items():
        base = RUNS[run_name]
        for f in folders:
            folder = base / f
            checks.append({
                "run": run_name,
                "required_item": f,
                "exists": folder.exists(),
                "file_count": len(list(folder.rglob("*"))) if folder.exists() else 0
            })

    return pd.DataFrame(checks)


# ============================================================
# Agent 3: Reproducibility Agent
# ============================================================

def reproducibility_agent():
    """
    Role:
    Creates a manifest of all generated tables, figures, reports, logs.
    """
    records = []

    for run_name, folder in RUNS.items():
        for file in find_files(folder):
            p = Path(file)
            records.append({
                "run": run_name,
                "file_name": p.name,
                "file_type": p.suffix,
                "full_path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 2)
            })

    df = pd.DataFrame(records)
    return df


# ============================================================
# Agent 4: Explainer Agent
# ============================================================

def explainer_agent():
    """
    Role:
    Produces paper-ready interpretation using only implemented results.
    """
    interpretation = {
        "main_finding": (
            "The implemented SEED-V experiments show that EEG biometric identity is not stationary. "
            "Verification performance changes under emotional, spectral, and session variability."
        ),
        "run02_interpretation": (
            "Cross-emotion analysis showed that cognitive-state transitions produce measurable identity drift, "
            "with mean EER around 0.0746 and mean AUC around 0.966."
        ),
        "run05_run06_interpretation": (
            "PSD and region-level analyses linked biometric drift to spectral and spatial neural changes, "
            "supporting biological interpretation rather than treating drift as only a machine-learning artifact."
        ),
        "run07_interpretation": (
            "Session-drift analysis showed degradation from S1 heldout EER 0.137977 to S1→S2 EER 0.181822 "
            "and S1→S3 EER 0.250755. Subject adaptation reduced S2 error by 11.44% and S3 error by 21.54%."
        ),
        "novelty_sentence": (
            "Existing EEG biometric studies primarily report aggregate verification performance, whereas this work "
            "explicitly models, quantifies, biologically validates, and adaptively mitigates cognitive-state-induced "
            "EEG identity drift."
        )
    }
    return interpretation


# ============================================================
# Agent 5: NVIDIA Platform Agent
# ============================================================

def nvidia_platform_agent():
    """
    Role:
    Maps your research pipeline to NVIDIA platform components.
    """
    platform = [
        {
            "component": "NVIDIA NeMo Agent Toolkit",
            "use": "Multi-agent orchestration for Scientist, Critic, Explainer, and Reproducibility agents."
        },
        {
            "component": "NVIDIA NIM / Nemotron",
            "use": "Reasoning layer for experiment planning, protocol auditing, and report generation."
        },
        {
            "component": "NVIDIA Triton Inference Server",
            "use": "Future deployment of trained EEG biometric verification models."
        },
        {
            "component": "TensorRT",
            "use": "Future model compression and low-latency EEG authentication inference."
        },
        {
            "component": "CUDA/cuDNN/PyTorch",
            "use": "GPU training and accelerated EEG feature/model computation."
        },
        {
            "component": "Qiskit/PennyLane",
            "use": "Optional quantum-classical biometric baselines."
        }
    ]
    return pd.DataFrame(platform)


# ============================================================
# Main Orchestration
# ============================================================

def main():
    print("=" * 80)
    print("RUN09: NVIDIA AGENTIC RESEARCH PLATFORM INTEGRATION")
    print("=" * 80)

    scientist = pd.DataFrame(scientist_agent())
    critic = critic_agent()
    reproducibility = reproducibility_agent()
    explanation = explainer_agent()
    nvidia_map = nvidia_platform_agent()

    scientist.to_csv(TAB / "scientist_agent_next_experiments.csv", index=False)
    critic.to_csv(TAB / "critic_agent_protocol_audit.csv", index=False)
    reproducibility.to_csv(TAB / "reproducibility_agent_manifest.csv", index=False)
    nvidia_map.to_csv(TAB / "nvidia_platform_mapping.csv", index=False)

    with open(TAB / "explainer_agent_interpretation.json", "w") as f:
        json.dump(explanation, f, indent=2)

    report = f"""# RUN09 NVIDIA Agentic Research Platform Integration Report

Generated: {datetime.now()}

## Purpose

This run integrates the implemented EEG biometric drift experiments with an NVIDIA-style agentic research framework.

The biometric evidence remains generated by the implemented SEED-V experiments.  
The NVIDIA agentic layer is used for experiment planning, protocol auditing, reproducibility checking, interpretation, and deployment planning.

## Implemented Agents

| Agent | Role |
|---|---|
| Scientist Agent | Suggests next valid experiments |
| Critic Agent | Checks protocol completeness and missing outputs |
| Reproducibility Agent | Creates manifest of generated files |
| Explainer Agent | Converts results into paper-ready interpretation |
| NVIDIA Platform Agent | Maps workflow to NeMo/NIM/Triton/TensorRT stack |

## Key Scientific Interpretation

{explanation["main_finding"]}

## RUN02 Interpretation

{explanation["run02_interpretation"]}

## RUN05/RUN06 Interpretation

{explanation["run05_run06_interpretation"]}

## RUN07 Interpretation

{explanation["run07_interpretation"]}

## Paper-Ready Novelty Sentence

{explanation["novelty_sentence"]}

## NVIDIA Platform Mapping

{nvidia_map.to_markdown(index=False)}

## Next Recommended Experiment

AEP cross-dataset validation should be implemented next to show external generalization.

"""

    report_path = REP / "run09_nvidia_agentic_research_platform_report.md"
    report_path.write_text(report)

    print("\n✅ RUN09 COMPLETE")
    print("Tables saved to:", TAB)
    print("Report saved to:", report_path)
    print("\nGenerated files:")
    for f in TAB.glob("*"):
        print(" -", f.name)
    print(" -", report_path.name)


if __name__ == "__main__":
    main()