#!/usr/bin/env python3
import re
import pandas as pd
from Bio import SeqIO

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
protein_fasta = "./data/ensembl/Homo_sapiens.GRCh38.pep.all.fa"
output_csv = "./data/ensembl/protein_sequences_qc.csv"

# -------------------------------------------------------------
# LOAD + PARSE
# -------------------------------------------------------------
records = []
for rec in SeqIO.parse(protein_fasta, "fasta"):
    parts = {
        k: v
        for k, v in (tok.split(":", 1) for tok in rec.description.split() if ":" in tok)
    }
    seq = str(rec.seq).strip().upper()

    if parts.get("gene_biotype") == "protein_coding":
        records.append({
            "gene_id": parts.get("gene") or parts.get("gene_id"),
            "gene_symbol": parts.get("gene_symbol") or parts.get("gene"),
            "transcript_id": parts.get("transcript") or parts.get("transcript_id"),
            "length": len(seq),
            "seq": seq,
        })


df = pd.DataFrame(records)
print(f"✅ Loaded {len(df):,} protein-coding transcripts")

# -------------------------------------------------------------
# QUALITY CHECKS
# -------------------------------------------------------------
print("\n--- Quality Checks ---")
print("Null values per column:\n", df.isnull().sum())
print("Empty sequences:", (df["seq"].str.len() == 0).sum())
print("Duplicate gene symbols:", df["gene_symbol"].duplicated().sum())

# Filter invalid amino acid sequences (only canonical 20 AAs)
df = df[df["seq"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)]
print(f"✅ Retained {len(df):,} valid protein sequences after QC")

# Save cleaned results
df.to_csv(output_csv, index=False)
print(f"💾 Saved cleaned table → {output_csv}")

print("\nSample:")
print(df.head(5).to_string(index=False))
