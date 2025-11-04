# app.py (multi-tenant)
import os, json
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from typing import Optional
import openai
import pinecone
from sheets_helpers import get_gsheet_client, read_sheet_as_df
import pandas as pd
from pathlib import Path

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ops-consultant")
GOOGLE_SA_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH")

TENANTS_JSON = os.getenv("TENANTS_JSON")
TENANTS_PATH = os.getenv("TENANTS_PATH", "./tenants.json")

if TENANTS_JSON:
    TENANTS = json.loads(TENANTS_JSON)
elif Path(TENANTS_PATH).exists():
    TENANTS = json.loads(Path(TENANTS_PATH).read_text())
else:
    TENANTS = {}

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no configurada")

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)

app = FastAPI(title="Ops-Consultant API (multi-tenant)")

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    question: str
    context_filters: Optional[dict] = None

def get_tenant_by_id(company_id: str):
    if not company_id:
        return None
    return TENANTS.get(company_id)

def verify_api_key_for_company(company_id: str, api_key: str):
    t = get_tenant_by_id(company_id)
    if not t:
        return False
    return t.get("api_key") == api_key

def get_sheet_id_for_company(company_id: str):
    t = get_tenant_by_id(company_id)
    if not t:
        return None
    return t.get("sheet_id")

def embed_query(text: str):
    res = openai.Embedding.create(model=OPENAI_EMBED_MODEL, input=text)
    return res["data"][0]["embedding"]

def pinecone_search(query: str, company_id: str, top_k=4):
    vec = embed_query(query)
    if company_id:
        qres = index.query(vector=vec, top_k=top_k, include_metadata=True, namespace=company_id)
    else:
        qres = index.query(vector=vec, top_k=top_k, include_metadata=True)
    items = []
    for m in qres.get("matches", []):
        items.append({
            "id": m["id"],
            "score": m["score"],
            "metadata": m.get("metadata")
        })
    return items

def build_data_summary_from_df(df: pd.DataFrame, filters: dict = None):
    if filters:
        for k,v in filters.items():
            if k in df.columns:
                df = df[df[k] == v]
    summary = {}
    if len(df)==0:
        return {"rows":0}
    for col in ["OEE", "Availability", "Performance", "Quality"]:
        if col in df.columns:
            try:
                summary[col] = float(df[col].mean())
            except:
                summary[col] = None
    if "Units OK" in df.columns and "Units KO" in df.columns:
        try:
            total_ok = int(df["Units OK"].sum())
            total_ko = int(df["Units KO"].sum())
            summary["Units OK"] = total_ok
            summary["Units KO"] = total_ko
            summary["Yield"] = total_ok / max(1, (total_ok+total_ko))
        except:
            pass
    summary["rows"] = len(df)
    return summary

def extract_tenant_info(request: Request):
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    company_id = request.headers.get("x-company-id") or request.headers.get("X-Company-Id")
    api_key = None
    if auth and auth.lower().startswith("bearer "):
        api_key = auth.split(" ",1)[1].strip()
    return company_id, api_key

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    company_id, api_key = extract_tenant_info(request)
    if not company_id or not api_key or not verify_api_key_for_company(company_id, api_key):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid company_id or api key")

    retrieved = pinecone_search(req.question, company_id=company_id, top_k=4)
    retrieved_text = ""
    for r in retrieved:
        meta = r["metadata"] or {}
        snippet = meta.get("text_snippet","")
        source = meta.get("filename", meta.get("source","doc"))
        retrieved_text += f"Fuente: {source}\nSnippet: {snippet}\n\n"

    data_summary = {}
    try:
        sheet_id = get_sheet_id_for_company(company_id)
        if not sheet_id:
            raise Exception("No sheet_id configured for this company")
        if not (GOOGLE_SA_PATH and sheet_id):
            raise Exception("Google Sheets config faltante")
        client = get_gsheet_client(GOOGLE_SA_PATH)
        df = read_sheet_as_df(client, sheet_id, "Company_X")
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='ignore')
            except:
                pass
        data_summary = build_data_summary_from_df(df, req.context_filters or {})
    except Exception as e:
        data_summary = {"error":"No se pudo leer Google Sheets: " + str(e)}

    system_prompt = (
        "Eres 'Consultor de Operaciones' para plantas de producción. Responde en español. "
        "Reglas: 1) Usa solo los datos proporcionados y los documentos recuperados. 2) Formato: Resumen corto, Hallazgos (con cifras), Recomendaciones accionables (prioritizadas), Nivel de confianza (Alta/Media/Baja), Referencias. "
        "3) Fórmulas: Availability=(Planned-Stop)/Planned; Performance=(Actual_output*Ideal_cycle_time)/Run_time; Quality=Good/Total; OEE=Availability*Performance*Quality."
    )

    user_prompt = f"""
Contexto de datos resumido (json):
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

Documentos recuperados (snippets):
{retrieved_text}

Pregunta del usuario:
{req.question}

Si necesitas más datos explícitos, pide exactamente la(s) columna(s) y periodo(s) que requieres del Google Sheet (ej. 'Necesito OEE diario de L1 entre 2025-05-01 y 2025-05-15').
"""

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_prompt}
            ],
            temperature=0.12,
            max_tokens=700
        )
        answer = resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error OpenAI: " + str(e))

    return {"answer": answer, "retrieved": retrieved, "data_summary": data_summary}

@app.get("/kpi/summary")
def kpi_summary(request: Request, filters: Optional[str]=None):
    company_id, api_key = extract_tenant_info(request)
    if not company_id or not api_key or not verify_api_key_for_company(company_id, api_key):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid company_id or api key")
    try:
        sheet_id = get_sheet_id_for_company(company_id)
        if not sheet_id:
            return {"error":"No sheet configured for this company"}
        client = get_gsheet_client(GOOGLE_SA_PATH)
        df = read_sheet_as_df(client, sheet_id, "Company_X")
        import json as _json
        fdict = _json.loads(filters) if filters else None
        summary = build_data_summary_from_df(df, fdict)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_doc")
async def ingest_doc(request: Request, file: UploadFile = File(...)):
    company_id, api_key = extract_tenant_info(request)
    if not company_id or not api_key or not verify_api_key_for_company(company_id, api_key):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid company_id or api key")

    import tempfile
    from ingest_docs_and_table import index_document_text
    content = await file.read()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    if suffix.lower() == ".pdf":
        from pdf_extract import extract_text_from_pdf
        text = extract_text_from_pdf(tmp_path)
    else:
        text = Path(tmp_path).read_text(encoding="utf-8")
    index_document_text(text, doc_id=file.filename, metadata={"filename": file.filename, "type":"methodology"}, company_id=company_id)
    return {"status":"ok", "filename": file.filename, "namespace": company_id}
