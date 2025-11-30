# Quantum Parenting Model (QPM) – Dialogue Analyzer

This project implements the **Quantum Parenting Model (QPM)**, a formalization of 
Baumrind’s parenting styles (authoritarian, permissive, authoritative) using a 
**Bloch-sphere computational framework**.  
The tool analyzes dialogue from films and Television (The Great Santini, Willy Wonka & the Chocolate Factory, and Leave It to Beaver)
and scores each line along the **QPM Bloch z-axis**:

- **+1 → permissive (|0⟩)**  
- **−1 → authoritarian (|1⟩)**  
- **0 → authoritative (balanced / superposition)**

The model learns a **log-odds polarity lexicon** from annotated transcripts of the film(s)/TV and applies it to new text, revealing how warmth and control oscillate across 
dialogue — the mathematical expression of Baumrind’s dialectic.

---

## Features

- Learns a permissive/authoritarian lexicon from movie, television transcripts  
- Sentence-level QPM scoring using a flipped log-odds + `tanh` model  
- Distinguishes parent vs. child dialogue using `[parent]` and `[child]` tags  
- Computes composite transcript score  
- Generates a Bloch z-axis polarity timeline  
- Clean Streamlit UI with expandable frames and styled blocks  
- Works with included transcripts and user-supplied dialogue  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/qpm-parenting.git
cd qpm-parenting

Install Dependencies:
pip install -r requirements.txt

## Running the App
streamlit run qpm.py

## File Structure
qpm-parenting/
├── qpm.py                        # Main Streamlit application
├── santini.txt                   # Authoritarian corpus (|1⟩)
├── wonka.txt                     # Permissive corpus (|0⟩)
├── beaver.txt                    # Authoritative example
├── qpm.png                       # Bloch sphere illustration
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

## Transcript Format
[parent] Yes sir, loud and clear.
[child] Okay, but I don't want to!
Lines without a tag default to "unknown".
Child lines receive a small positive weighting boost to reflect system response.

## Citation
If used academically, please cite the corresponding paper:
Rodriguez, M. C. (2025). Baumrind’s Parenting Dialectic: Santini, Veruca, and the Beaver.
Quantum Parenting Model (QPM) implementation.

## License
This project is licensed under the MIT License.
See the LICENSE file for details.



