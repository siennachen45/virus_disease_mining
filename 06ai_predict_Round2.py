
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

SYSTEM_PROMPT = """You are a rigorous biomedical expert. Your task is to determine whether the provided sentence supports a DIRECT pathogenic, causal, or specific clinical-syndromic association between a VIRUS and a TARGET DISEASE.

You must apply STRICT disease-name matching and causality validation.

Strict Validation Criteria
==========================

Return True ONLY if BOTH conditions are satisfied:
(A) Direct pathogenic / causal association is explicitly stated
AND
(B) The caused condition clearly matches the TARGET DISEASE itself

If either condition fails, return False.

-----------------------------------
1. TRUE (Confirm Direct Association)
-----------------------------------

Return True only when the sentence explicitly confirms that the virus directly causes, induces, triggers, or is clinically associated with the TARGET DISEASE itself.

Valid positive cases include:

- Named virus-specific syndromes:
  e.g. "HIV-associated neuropathy", "EBV-associated lymphoma"

- Explicit causal wording:
  "virus-induced", "virus-caused", "virus-triggered", "secondary to viral infection"

- Direct clinical complications:
  organ-specific, neurological, vascular, hematological, systemic, behavioral, or syndrome-level complications directly attributed to the virus

- Manifestations / sequelae:
  symptoms, syndromes, or recognized disease entities directly resulting from viral pathology

IMPORTANT:
The supported disease in the sentence must MATCH the TARGET DISEASE in accepted clinical meaning.

Matching includes:
- exact disease name
- standard synonym
- MeSH synonym / entry term
- widely accepted clinical alias
- standard narrower syndrome only when explicitly equivalent

-----------------------------------
2. FALSE (Reject Association)
-----------------------------------

Return False in ANY of the following cases.

A. Disease mismatch (VERY IMPORTANT)
-----------------------------------

This is the most important rejection rule.

Even if the sentence contains strong causal phrases such as:
- virus-induced
- infectious trigger
- virus-associated
- due to viral infection

you MUST return False unless the caused condition matches the TARGET DISEASE.

Reject if the sentence supports:
- another disease
- another syndrome
- another complication
- another lesion
- another anatomical localization
- only a broader disease category
- only a narrower but non-equivalent subtype
- a related disease in the same organ system

Examples of mismatch:
- "acute vascular disorder" ≠ "Cerebrovascular Disorders"
- "CNS lesions" ≠ "Basal Ganglia Diseases"
- "neurological complications" ≠ a specific CNS syndrome
- "vascular inflammation" ≠ stroke / cerebrovascular disease

Same-organ-system proximity does NOT count.

A general vascular disorder does NOT imply cerebrovascular disease.
A general CNS lesion does NOT imply basal ganglia disease.

Do NOT infer based on anatomical similarity alone.

B. Broad / vague descriptions
-----------------------------

Reject vague, generic, or non-specific descriptions.

Examples:
- broad mention of viruses as clinical targets
- general disease burden statements
- epidemiological summaries
- speculative wording
- possible association without explicit causality

Phrases like:
- "may be involved"
- "potential role"
- "emerging evidence"
- "associated with inflammatory process"

are NOT sufficient unless the TARGET DISEASE is explicitly matched.

C. Background condition / comorbidity
-------------------------------------

The virus is only background status:
e.g. "HIV-positive patient"

but the disease is caused by trauma, surgery, congenital injury, radiation, treatment, etc.

D. Secondary opportunistic infection
------------------------------------

The virus causes immunosuppression, but the actual disease is caused by another pathogen.

Return False.

E. Experimental / molecular findings
------------------------------------

Reject molecular mechanisms, pathways, biomarkers, or cell-level findings.

Examples:
- DNA damage
- oxidative stress
- apoptosis
- gene upregulation
- inflammation markers

These are NOT disease associations unless they explicitly correspond to the TARGET DISEASE.

F. Vector / biotechnology use
-----------------------------

If virus terms such as:
- AAV
- adenovirus
- lentivirus
- retrovirus

are used as vectors for gene delivery, expression, overexpression, experimental manipulation, or therapy,
return False.

-----------------------------------
3. Core Decision Rule
-----------------------------------

DO NOT judge based only on strong causal wording.

You must verify BOTH:
1. direct virus → disease causality
2. exact clinical correspondence to TARGET DISEASE

If the sentence proves causality but for the wrong disease,
return False.

When uncertain, prefer False.

-----------------------------------
Output
-----------------------------------

Return ONLY:
"True"
or
"False"
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/pair_groups_for_llm_review.xlsx")
    parser.add_argument("--out", default=None)
    parser.add_argument("--id_col", default="pair_group_id")
    parser.add_argument("--virus_col", default="virus")
    parser.add_argument("--disease_col", default="disease")
    parser.add_argument("--sentence_col", default="sentence")
    parser.add_argument("--label_col", default="llm_label")
    parser.add_argument("--max_sentences", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--model", default=os.environ.get("MOONSHOT_MODEL_NAME", "moonshot-v1-8k"))
    parser.add_argument("--base_url", default=os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"))
    return parser.parse_args()


def default_output_path(input_path):
    path = Path(input_path)
    return path.with_name(f"{path.stem}_llm_reviewed{path.suffix}")


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t")


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def first_non_empty(values):
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def get_llm_label(client, model, virus, disease, sentences, max_retries):
    content = f"Virus: {virus}\nDisease: {disease}\nSentences to analyze:\n"
    for i, sentence in enumerate(sentences):
        content += f"[{i + 1}] {sentence}\n"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                timeout=30,
            )
            answer = completion.choices[0].message.content.strip().lower()
            if "true" in answer:
                return "True"
            if "false" in answer:
                return "False"
            return "Unknown"
        except Exception as exc:
            print(f"  [Error] Attempt {attempt + 1} failed: {exc}")
            time.sleep(2)
    return "Error"


def main():
    args = parse_args()
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("Set MOONSHOT_API_KEY before running this script.")

    output_path = args.out or default_output_path(args.input)
    df = read_table(args.input)
    if args.label_col not in df.columns:
        df[args.label_col] = None

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    groups = df.groupby(args.id_col, sort=False)
    print(f"Total groups: {len(groups)}")

    processed = 0
    try:
        for group_id, group_df in groups:
            first_idx = group_df.index[0]
            existing_label = str(df.at[first_idx, args.label_col])
            if existing_label in ["True", "False"]:
                continue

            virus = first_non_empty(group_df[args.virus_col])
            disease = first_non_empty(group_df[args.disease_col])
            sentences = [str(s) for s in group_df[args.sentence_col].dropna().tolist()]
            sentences = sentences[: args.max_sentences]

            print(f"Processing group {group_id} ({virus} - {disease})...", end="", flush=True)
            label = get_llm_label(client, args.model, virus, disease, sentences, args.max_retries)
            df.at[first_idx, args.label_col] = label
            write_table(df, output_path)

            processed += 1
            print(f" -> Result: {label}")
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[Interrupted] Progress saved.")

    print(f"\nFinished. Processed {processed} new groups.")


if __name__ == "__main__":
    main()
