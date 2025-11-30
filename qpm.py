import math
import re
from pathlib import Path
from collections import Counter

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

# ---------------------------
# Utility: load and parse transcripts
# ---------------------------
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 75% !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("qpm.png", caption="Quantum Parenting Model (QPM) – Authoritarian vs Permissive Dialogue Analyzer", width=1024)

SPEAKER_PATTERN = re.compile(r'^\s*\[(?P<role>parent|child|children)\]\s*(?P<text>.*)$', re.IGNORECASE)

def parse_transcript(path: Path):
    """
    Parse a transcript file into a list of dicts:
    [{'speaker': 'parent'/'child'/'children', 'text': '...'}, ...]
    Lines without a tag are treated as 'unknown'.
    """
    lines = []
    if not path.exists():
        return lines

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            m = SPEAKER_PATTERN.match(raw)
            if m:
                role = m.group("role").lower()
                text = m.group("text").strip()
            else:
                role = "unknown"
                text = raw
            lines.append({"speaker": role, "text": text})
    return lines

# ---------------------------
# Tokenization / preprocessing
# ---------------------------

STOPWORDS = {
    "the", "and", "a", "an", "to", "of", "in", "it", "is", "on", "that", "this",
    "for", "with", "as", "you", "i", "we", "they", "he", "she", "be", "are",
    "was", "were", "at", "by", "or", "do", "did", "so", "but", "if", "then",
    "from", "your", "my", "me", "our", "us", "him", "her", "them", "up", "out"
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")

def tokenize(text: str):
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

# ---------------------------
# Build lexicon from corpora
# ---------------------------

def build_lexicon(authoritarian_lines, permissive_lines, min_freq=2, smoothing=1.0):
    """
    Build a word-level lexicon using log-odds weights:
    weight(w) > 0 => authoritarian leaning
    weight(w) < 0 => permissive leaning
    """
    auth_counts = Counter()
    perm_counts = Counter()

    for item in authoritarian_lines:
        tokens = tokenize(item["text"])
        auth_counts.update(tokens)

    for item in permissive_lines:
        tokens = tokenize(item["text"])
        perm_counts.update(tokens)

    total_auth = sum(auth_counts.values()) + smoothing * len(auth_counts)
    total_perm = sum(perm_counts.values()) + smoothing * len(perm_counts)

    weights = {}
    vocab = set(auth_counts.keys()) | set(perm_counts.keys())

    for w in vocab:
        ca = auth_counts.get(w, 0) + smoothing
        cp = perm_counts.get(w, 0) + smoothing
        p_auth = ca / total_auth
        p_perm = cp / total_perm
        weight = math.log(p_auth) - math.log(p_perm)
        # Only keep words that appear at least min_freq times across both corpora
        if ca + cp >= min_freq:
            weights[w] = weight

    return weights, auth_counts, perm_counts

# ---------------------------
# Sentence scoring
# ---------------------------

def score_sentence(text: str, speaker: str, weights: dict, child_boost: float = 1.2):
    """
    Returns QPM score in [-1, 1] with Bloch-aligned mapping:
    +1 -> |0⟩ = permissive (north pole)
    -1 -> |1⟩ = authoritarian (south pole)
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    raw = 0.0
    for t in tokens:
        raw += weights.get(t, 0.0)

    # raw > 0 means more authoritarian words than permissive
    # Compress to [-1, 1] using tanh
    score = math.tanh(raw)

    # Boost child responses, since they reflect system dynamics
    speaker = speaker.lower()
    if speaker in {"child", "children"} and score != 0.0:
        score *= child_boost

    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))

    score = -score

    return score

def score_transcript(lines, weights):
    scored = []
    for idx, item in enumerate(lines, start=1):
        s = score_sentence(item["text"], item["speaker"], weights)
        scored.append({
            "line": idx,
            "speaker": item["speaker"],
            "text": item["text"],
            "qpm_score": s
        })
    return scored

# ---------------------------
# Streamlit app
# ---------------------------

def main():
    # Combined CSS for cleanliness
    st.markdown(
    """
    <style>
        .qpm-green {
            color: #0b8f36 !important;
        }
        .big-text {
            font-size: 26px !important;
            font-weight: 500;
            line-height: 1.4;
        }
    </style>
    """,
    unsafe_allow_html=True
)


    # Main content block (fixed HTML structure)
    st.markdown(
        """
        <p class="big-text">
            This program constructs a <strong>parenting-style lexicon</strong> from two movie scenes representing the QPM basis states:
        </p>
        
        <div class="big-text">
            • <strong>Willy Wonka & the Chocolate Factory – Veruca Salt’s  
            <a href="https://youtu.be/b9mba2qb9do?si=hoPUXLktC6_euuKq" target="_blank">Golden Ticket</a></strong><br>
            <span class="qpm-green">(permissive parenting basis; mapped to |0⟩, Bloch Sphere North pole, z = +1)</span>
        </div>

        <br>

        <div class="big-text">
            • <strong>The Great Santini – 
            <a href="https://youtu.be/Pj2K4FrqTmw?si=NY0_o1THGuRBQcI3" target="_blank">Marine Kids</a></strong><br>
            <span class="qpm-green">(authoritarian parenting basis; mapped to |1⟩, Bloch Sphere South pole, z = -1)</span>
        </div>

        <br>        

        <p class="big-text">
            Using this lexicon, the model scores each line of dialogue on a scale from 
            <strong>−1 (authoritarian / |1⟩)</strong> to 
            <strong>+1 (permissive / |0⟩)</strong>.
        </p>

        <p class="big-text">
            Child responses are given a slight weighting boost to reflect the structure—or chaos—of the parenting system they occur in.
        </p>
        """,
        unsafe_allow_html=True
    )


    base_path = Path(__file__).parent
    santini_path = base_path / "santini.txt"
    wonka_path = base_path / "wonka.txt"
    beaver_path = base_path / "beaver.txt"

    # Load corpora
    santini_lines = parse_transcript(santini_path)
    wonka_lines = parse_transcript(wonka_path)
    beaver_lines = parse_transcript(beaver_path)

    # Build lexicon
    if not santini_lines or not wonka_lines:
        st.error("Could not find or parse 'santini.txt' and 'wonka.txt'. "
                 "Please ensure they are in the same folder as this app.")
        return

    weights, auth_counts, perm_counts = build_lexicon(santini_lines, wonka_lines, min_freq=2, smoothing=1.0)

    st.sidebar.header("Analysis Options")

    corpus_choice = st.sidebar.selectbox(
        "Choose a transcript to analyze:",
        ("Santini (authoritarian scene)", "Wonka (permissive scene)",
         "Beaver (authoritative scene)", "Custom text")
    )

    if corpus_choice == "Santini (authoritarian scene)":
        lines = santini_lines
        st.subheader("Analyzing: The Great Santini (\"Marine Kids\" scene)")
    elif corpus_choice == "Wonka (permissive scene)":
        lines = wonka_lines
        st.subheader("Analyzing: Willy Wonka – Veruca \"I Want It Now\" scene")
    elif corpus_choice == "Beaver (authoritative scene)":
        if not beaver_lines:
            st.error("Could not find or parse 'beaver.txt'.")
            return
        lines = beaver_lines
        st.subheader("Analyzing: Leave It to Beaver (authoritative example)")
    else:
        st.subheader("Custom Transcript")
        st.markdown(
            "Paste lines below. Use `[parent]`, `[child]`, or `[children]` at the start of each line to mark speakers."
        )
        custom_text = st.text_area("Custom transcript", height=200, value="[parent] Example line here.")
        if custom_text.strip():
            lines = []
            for raw in custom_text.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                m = SPEAKER_PATTERN.match(raw)
                if m:
                    role = m.group("role").lower()
                    text = m.group("text").strip()
                else:
                    role = "unknown"
                    text = raw
                lines.append({"speaker": role, "text": text})
        else:
            lines = []

    if not lines:
        st.info("No lines available to score.")
        return

    scored = score_transcript(lines, weights)
    df = pd.DataFrame(scored)

    # Composite score
    composite = df["qpm_score"].mean()
    st.markdown(
    f"""
    <p class="big-text">
        <strong>Composite QPM score for this transcript:</strong> 
        <code>{composite:.3f}</code>
    </p>

    <p class="big-text">
        <em>Interpretation</em>:  
        −1 ≈ authoritarian (|1⟩),  
        +1 ≈ permissive (|0⟩),  
        0 ≈ authoritative balance (equator).
    </p>
    """,
    unsafe_allow_html=True
)
    st.markdown("### Sentence-by-sentence scores")
    st.dataframe(df[["line", "speaker", "qpm_score", "text"]], use_container_width=False)
    
    # Timeline plot
    # ---- Create the figure ----
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.plot(df["line"], df["qpm_score"], marker="o")
    ax.axhline(0.0, linestyle="--")
    
    # Explicit font sizes
    ax.set_xlabel("Line number", fontsize=10)
    ax.set_ylabel(
    	"QPM Bloch z-score\n(-1 = |1⟩ authoritarian, +1 = |0⟩ permissive)",
    	fontsize=10
    )
    ax.set_title(
    	"Authoritarian–Permissive Polarity Over Dialogue",
    	fontsize=12
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_ylim(-1.05, 1.05)
    
    # ---- Convert figure to PNG in memory ----
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    
    # ---- Show framed chart with exact width ----
    st.markdown(
    	f"""
    	<div style="
    		width: 1000px;                 /* set exact frame width */
    		margin-left: 0;
    		margin-right: auto;
    		padding: 15px;
    		border: 1px solid #ccc;
    		border-radius: 8px;
    		background-color: #ffffff;
    		text-align: center;
    	">
    		<img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto;" />
    	</div>
    	""",
    	unsafe_allow_html=True
    )

    # Show top lexicon entries for curiosity
    st.markdown("### Learned Lexicon (most indicative words)")

    # Sort by weight (note: weights > 0 = authoritarian, < 0 = permissive *before* flip)
    sorted_words = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_auth = sorted_words[:15]
    top_perm = sorted(weights.items(), key=lambda x: x[1])[:15]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most authoritarian-leaning words (basis for |1⟩)**")
        st.table(pd.DataFrame(top_auth, columns=["word", "weight"]))
    with col2:
        st.markdown("**Most permissive-leaning words (basis for |0⟩)**")
        st.table(pd.DataFrame(top_perm, columns=["word", "weight"]))
    
    st.markdown(
        """
<p class="big-text">
        **INTERPRETATION**
</p><p class="big-text">
        Scores near <strong>+1</strong> indicate communication dominated by the permissive basis <strong>|0⟩</strong> — warm, indulgent, low structure.
</p>

<p class="big-text">
    Scores near <strong>−1</strong> indicate communication dominated by the authoritarian basis <strong>|1⟩</strong> — rigid, controlling, and highly structured.
</p>

<p class="big-text">
    Scores near <strong>0</strong> indicate an authoritative <strong>superposition</strong>, where warmth and structure are balanced — the Bloch equator.
</p>

<p class="big-text">
    In QPM formalism, this corresponds to the expectation value of the parenting-style operator  
    <strong><code>Ẑ<sub>Parent</sub></code></strong>, where:
</p>

<p class="big-text qpm-green">
    <code>Ẑ<sub>Parent</sub>|0⟩ = +1|0⟩</code> (permissive) &nbsp;&nbsp;&nbsp;
    <code>Ẑ<sub>Parent</sub>|1⟩ = −1|1⟩</code> (authoritarian)
</p>

<p class="big-text">
    Thus, the Bloch z-axis tracks how a dialogue oscillates between these poles across turns, revealing the dynamic balance of warmth and control communicated in the interaction.
</p>

        """,
        unsafe_allow_html=True
    )




if __name__ == "__main__":
    main()


