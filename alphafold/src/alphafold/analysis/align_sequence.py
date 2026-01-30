from pathlib import Path
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices


def generate_alignment_file(
    query_seq: str,
    target_seq: str,
    query_name: str,
    target_name: str,
    filename: str = "alignment.fa",
) -> str:
    """Generates a pairwise alignment file in FASTA format with custom names."""
    q_rec = SeqRecord(Seq(query_seq), id=query_name, description="")
    t_rec = SeqRecord(Seq(target_seq), id=target_name, description="")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(q_rec, t_rec)
    best_alignment = alignments[0]

    alignment_content = best_alignment.format("fasta")

    alignment_path = Path(filename)
    alignment_path.write_text(alignment_content)

    return str(alignment_path.absolute())
