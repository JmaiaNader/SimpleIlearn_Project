"""InsightForge data loading, business analytics, and lightweight RAG retrieval."""
from pathlib import Path
import json
import zipfile
import pandas as pd

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

SUPPORTED_TABLES = {".csv", ".xlsx", ".xls", ".json"}
SUPPORTED_DOCS = {".pdf", ".txt", ".md"}

def _read_table(path: Path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
    except Exception as exc:
        print(f"Skipping {path.name}: {exc}")
    return None

def _extract_pdf(path: Path):
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def load_dataset(source_path):
    """Load supported files from a folder or ZIP. Returns (tables, documents, messages)."""
    source = Path(source_path).expanduser()
    tables, documents, messages = [], [], []
    if not source.exists():
        return tables, documents, [f"Path not found: {source}"]

    files = []
    if source.is_file() and source.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source) as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    suffix = Path(name).suffix.lower()
                    if suffix not in SUPPORTED_TABLES | SUPPORTED_DOCS:
                        continue
                    try:
                        raw = zf.read(name)
                        if suffix == ".csv":
                            import io
                            df = pd.read_csv(io.BytesIO(raw))
                            tables.append({"name": Path(name).name, "path": name, "data": df})
                        elif suffix in {".xlsx", ".xls"}:
                            import io
                            df = pd.read_excel(io.BytesIO(raw))
                            tables.append({"name": Path(name).name, "path": name, "data": df})
                        elif suffix == ".json":
                            import io
                            df = pd.read_json(io.BytesIO(raw))
                            tables.append({"name": Path(name).name, "path": name, "data": df})
                        elif suffix == ".pdf" and PdfReader:
                            import io
                            reader = PdfReader(io.BytesIO(raw))
                            body = "\n".join(p.extract_text() or "" for p in reader.pages)
                            documents.append({"name": Path(name).name, "text": body})
                        elif suffix in {".txt", ".md"}:
                            documents.append({"name": Path(name).name, "text": raw.decode("utf-8", errors="ignore")})
                    except Exception as exc:
                        messages.append(f"Could not read {name}: {exc}")
        except Exception as exc:
            messages.append(f"Could not open ZIP: {exc}")
    else:
        files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
        for path in files:
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_TABLES:
                df = _read_table(path)
                if df is not None:
                    tables.append({"name": path.name, "path": str(path), "data": df})
            elif suffix == ".pdf":
                body = _extract_pdf(path)
                if body:
                    documents.append({"name": path.name, "text": body})
            elif suffix in {".txt", ".md"}:
                try:
                    documents.append({"name": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")})
                except Exception:
                    pass

    if not tables and not documents:
        messages.append("No readable CSV, Excel, JSON, PDF, TXT, or Markdown files were found.")
    return tables, documents, messages

def dataframe_profile(df):
    """Return useful, compact statistics without assuming specific column names."""
    profile = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "missing_values": int(df.isna().sum().sum()),
        "numeric_summary": {},
        "categorical_summary": {},
    }
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns[:12]:
        series = numeric[col].dropna()
        if len(series):
            profile["numeric_summary"][str(col)] = {
                "mean": round(float(series.mean()), 3),
                "median": round(float(series.median()), 3),
                "std": round(float(series.std()), 3) if len(series) > 1 else 0,
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
                "sum": round(float(series.sum()), 3),
            }
    for col in df.select_dtypes(exclude="number").columns[:8]:
        values = df[col].dropna().astype(str).value_counts().head(8)
        if len(values):
            profile["categorical_summary"][str(col)] = {str(k): int(v) for k, v in values.items()}
    return profile

def make_retrieval_chunks(tables, documents, chunk_chars=900):
    chunks = []
    for item in tables:
        df = item["data"]
        profile = dataframe_profile(df)
        chunks.append({
            "source": item["name"],
            "text": f"Dataset: {item['name']}\nProfile:\n{json.dumps(profile, ensure_ascii=False)}"
        })
        # Include sample rows so retrieval can find column/value references.
        sample = df.head(30).to_string(index=False)
        if sample:
            chunks.append({"source": item["name"], "text": f"Sample rows from {item['name']}:\n{sample}"})
    for doc in documents:
        text = doc["text"]
        for start in range(0, max(len(text), 1), chunk_chars):
            part = text[start:start + chunk_chars].strip()
            if part:
                chunks.append({"source": doc["name"], "text": part})
    return chunks

def retrieve_context(question, chunks, top_k=4):
    """Simple local lexical retrieval. Keeps the project dependency-light."""
    if not chunks:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [chunks[i] for i in ranked if scores[i] > 0]
    except Exception:
        terms = set(question.lower().split())
        ranked = sorted(chunks, key=lambda c: sum(t in c["text"].lower() for t in terms), reverse=True)
        return ranked[:top_k]

def quick_answer(question, tables, documents, retrieved):
    """Fallback answer with grounded dataset facts when no LLM is available."""
    parts = []
    q = question.lower()
    for item in tables:
        df, name = item["data"], item["name"]
        parts.append(f"Dataset **{name}** contains {len(df):,} rows and {len(df.columns)} columns.")
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            col = numeric.columns[0]
            parts.append(f"For **{col}**, mean = {numeric[col].mean():,.3f}, median = {numeric[col].median():,.3f}, standard deviation = {numeric[col].std():,.3f}.")
        date_cols = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]
        if date_cols:
            parts.append(f"Potential time field(s): {', '.join(map(str, date_cols[:4]))}. Use these for time-period analysis after confirming their format.")
        if any(word in q for word in ["region", "product", "category", "segment", "customer", "demographic", "sales"]):
            for col in df.select_dtypes(exclude="number").columns:
                if any(word in str(col).lower() for word in ["region", "product", "category", "segment", "customer", "gender", "age", "sales"]):
                    top = df[col].astype(str).value_counts().head(5).to_dict()
                    parts.append(f"Most common values in **{col}**: {top}")
    if documents:
        parts.append(f"Loaded {len(documents)} reference document(s) for retrieval.")
    if retrieved:
        parts.append("Relevant retrieved context:\n" + "\n\n".join(f"[{x['source']}] {x['text'][:700]}" for x in retrieved))
    return "\n\n".join(parts) if parts else "No readable data was loaded. Check the source path and supported file types."
