#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""OTU clustering"""

import argparse
import sys
import os
import gzip
import statistics
import textwrap
from pathlib import Path
from collections import Counter
from typing import Iterator, Dict, List
# https://github.com/briney/nwalign3
# ftp://ftp.ncbi.nih.gov/blast/matrices/
import nwalign3 as nw
import numpy as np
np.int = int

__author__ = "Souad Youjil Abadi"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Souad Youjil Abadi"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Souad Youjil Abadi"
__email__ = "souad_youjil@hotmail.com"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', '-amplicon_file', dest='amplicon_file', type=isfile, required=True,
                        help="Amplicon is a compressed fasta file (.fasta.gz)")
    parser.add_argument('-s', '-minseqlen', dest='minseqlen', type=int, default = 400,
                        help="Minimum sequence length for dereplication (default 400)")
    parser.add_argument('-m', '-mincount', dest='mincount', type=int, default = 10,
                        help="Minimum count for dereplication  (default 10)")
    parser.add_argument('-o', '-output_file', dest='output_file', type=Path,
                        default=Path("OTU.fasta"), help="Output file")
    return parser.parse_args()


def read_fasta(amplicon_file: Path, minseqlen: int) -> Iterator[str]:
    """Read a compressed fasta and extract all fasta sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :return: A generator object that provides the Fasta sequences (str).
    """
    with gzip.open(amplicon_file, 'rt') as f:
        # Initialize an empty string to store the current sequence
        seq = ''
        # Iterate over each line in the file
        for line in f:
            # Check if the line starts with the '>' character,
            # which indicates the start of a new sequence
            if line.startswith('>'):
                # If the length of the previous sequence is >=
                # to the minimum sequence length, yield the sequence
                if len(seq) >= minseqlen:
                    yield seq
                seq = ''
            else:
                # Append the current line to the sequence variable,
                # removing leading and trailing whitespace
                seq += line.strip()
        # If the length of the last sequence is >=
        # to the minimum sequence length, yield the sequence
        if len(seq) >= minseqlen:
            yield seq


def dereplication_fulllength(amplicon_file: Path, minseqlen: int, mincount: int) -> Iterator[List]:
    """Dereplicate the set of sequence

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :param mincount: (int) Minimum amplicon count
    :return: A generator object that provides a (list)[sequences, count] of
    sequence with a count >= mincount and a length >= minseqlen.
    """
    # Read the FASTA file and extract the sequences
    sequences = read_fasta(amplicon_file, minseqlen)
    # Count the occurrences of each sequence
    counts = Counter(sequences)
    # Yield the sequences with a count >= to the minimum count
    # and a length >= to the minimum sequence length
    for seq, count in counts.most_common():
        if count >= mincount and len(seq) >= minseqlen:
            yield [seq, count]


def get_identity(alignment_list: List[str]) -> float:
    """Compute the identity rate between two sequences

    :param alignment_list:  (list) A list of aligned sequences in the format 
    ["SE-QUENCE1", "SE-QUENCE2"]
    :return: (float) The rate of identity between the two sequences.
    """
    # Calculate the number of identical nucleotides in the alignment
    num_identical = sum(1 for a, b in zip(alignment_list[0], alignment_list[1]) if a == b)
    # Identity rate = number of identical nucleotides / length of the alignment
    identity_rate = num_identical / len(alignment_list[0]) * 100
    # Return the identity rate
    return identity_rate


def abundance_greedy_clustering(amplicon_file: Path, minseqlen: int, mincount: int, chunk_size: int = 50, kmer_size: int = 50) -> List:
    """Compute an abundance greedy clustering regarding sequence count and identity.
    Identify OTU sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :param mincount: (int) Minimum amplicon count.
    :param chunk_size: (int) A fournir mais non utilise cette annee
    :param kmer_size: (int) A fournir mais non utilise cette annee
    :return: (list) A list of all the [OTU (str), count (int)] .
    """
    # Dereplicate the sequences
    dereplicated_seqs = dereplication_fulllength(amplicon_file, minseqlen, mincount)
    # Initialize a list to store the OTUs
    otus = []
    # Iterate over the dereplicated sequences
    for seq, count in dereplicated_seqs:
        # Flag to indicate if the sequence is similar to an existing OTU
        similar_to_otu = False
        # Iterate over the existing OTUs
        for otu in otus:
            # Align the sequence with the OTU using global alignment
            alignment = nw.global_align(seq, otu[0])
            # Calculate the identity rate between the sequence and the OTU
            identity_rate = get_identity(alignment)
            # If identity rate is >= to 97, the sequence is similar to the OTU
            if identity_rate >= 97:
                # Add the count of the sequence to the count of the OTU
                otu[1] += count
                # Flag indicating the sequence is similar to an existing OTU
                similar_to_otu = True
                # Break out of the loop since we've found a match
                break
        # If the sequence is not similar to any existing OTUs,
        # add it as a new OTU
        if not similar_to_otu:
            otus.append([seq, count])
    # Sort the OTUs in descending order of their counts
    otus.sort(key=lambda x: x[1], reverse=True)
    # Return the list of OTUs
    return otus


def write_OTU(OTU_list: List, output_file: Path) -> None:
    """Write the OTU sequence in fasta format.

    :param OTU_list: (list) A list of OTU sequences
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as f:
        for i, (seq, count) in enumerate(OTU_list, start=1):
            # Generate the OTU header
            header = f">OTU_{i} occurrence:{count}\n"
            # Wrap the sequence to 80 characters per line
            wrapped_seq = textwrap.fill(seq, width=80)
            f.write(header + wrapped_seq + "\n")


#==============================================================
# Main program
#==============================================================
def main(): # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    # Compute the OTUs
    otus = abundance_greedy_clustering(args.amplicon_file, args.minseqlen, args.mincount)
    # Write the OTUs to the output file
    write_OTU(otus, args.output_file)


if __name__ == '__main__':
    main()
