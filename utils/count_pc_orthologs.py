from pathlib import Path

protein_csv = Path("protein_coding_genes.csv")
ortholog_file = Path("orthologs_one2one.txt")

# -------------------------------------------------
# 1. Load protein-coding gene symbols
# -------------------------------------------------
pc_genes = set()
with open(protein_csv) as f:
    next(f)  # header
    for line in f:
        symbol = line.split(",")[0].strip().upper()
        if symbol:
            pc_genes.add(symbol)

# -------------------------------------------------
# 2. Scan ortholog table
# -------------------------------------------------
all_human_genes = []
pc_matches = []
non_pc_matches = []

with open(ortholog_file) as f:
    next(f)  # header
    for line in f:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 5:
            continue

        human_gene = fields[4].strip().upper()
        if not human_gene:
            continue

        all_human_genes.append(human_gene)

        if human_gene in pc_genes:
            pc_matches.append(human_gene)
        else:
            non_pc_matches.append(human_gene)

# -------------------------------------------------
# 3. Summaries
# -------------------------------------------------
all_unique = set(all_human_genes)
pc_unique = set(pc_matches)
non_pc_unique = set(non_pc_matches)

print("\n===== SUMMARY =====")
print(f"Total ortholog rows: {len(all_human_genes):,}")
print(f"Unique human ortholog genes: {len(all_unique):,}")
print(f"Protein-coding ortholog genes (unique): {len(pc_unique):,}")
print(f"Non-protein-coding ortholog genes (unique): {len(non_pc_unique):,}")

if all_unique:
    print(f"Percent protein-coding: {100*len(pc_unique)/len(all_unique):.2f}%")

# -------------------------------------------------
# 4. Show quick examples of non-protein-coding genes
# -------------------------------------------------
print("\nExamples of non-protein-coding ortholog genes:")
for g in sorted(non_pc_unique)[:10]:
    print("  ", g)

# -------------------------------------------------
# 5. Optional: save lists
# -------------------------------------------------
#Path("ortholog_pc_genes.txt").write_text("\n".join(sorted(pc_unique)))
#Path("ortholog_non_pc_genes.txt").write_text("\n".join(sorted(non_pc_unique)))

#print("\nSaved:")
#print("  ortholog_pc_genes.txt")
#print("  ortholog_non_pc_genes.txt")
