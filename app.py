from fastapi import FastAPI, Form, Request, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import aiofiles
import csv
import asyncio

from typing import List
from pypdf import PdfReader
from fpdf import FPDF

from src.helper import llm_pipeline

app = FastAPI()

# CORS (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure dirs
os.makedirs("static/docs", exist_ok=True)
os.makedirs("static/output", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def _safe_pdf_name(filename: str) -> str:
    name = os.path.basename(filename)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")
    return name

def _count_pdf_pages(path: str) -> int:
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception:
        return -1

@app.post("/upload")
async def upload_pdf(pdf_file: bytes = File(...), filename: str = Form(...)):
    try:
        base_folder = "static/docs"
        os.makedirs(base_folder, exist_ok=True)

        safe_name = _safe_pdf_name(filename)
        pdf_path = os.path.join(base_folder, safe_name)

        async with aiofiles.open(pdf_path, "wb") as f:
            await f.write(pdf_file)

        # enforce max 5 pages
        pages = _count_pdf_pages(pdf_path)
        if pages != -1 and pages > 5:
            try:
                os.remove(pdf_path)
            except Exception:
                pass
            return JSONResponse(
                status_code=200,
                content={"msg": "error", "reason": "max_pages_exceeded", "pages": pages},
            )

        return JSONResponse({"msg": "success", "pdf_filename": pdf_path})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def _write_csv(output_file: str, rows: List[List[str]]):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "Answer"])
        for q, a in rows:
            a_clean = " ".join(str(a).split())
            writer.writerow([q, a_clean])

def _write_pdf(output_file: str, rows: List[List[str]]):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Using default core fonts to avoid external TTF requirement
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Interview Q&A", ln=True)
    pdf.ln(3)
    for i, (q, a) in enumerate(rows, start=1):
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 8, f"Q{i}. {q}")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, f"Ans: {a}")
        pdf.ln(2)
    pdf.output(output_file)

def generate_outputs_sync(file_path: str, model_name: str):
    answer_generation_chain, ques_list = llm_pipeline(file_path, model_name=model_name)

    base_folder = "static/output"
    os.makedirs(base_folder, exist_ok=True)

    base = os.path.splitext(os.path.basename(file_path))[0]
    csv_path = os.path.join(base_folder, f"{base}_QA.csv")
    pdf_path = os.path.join(base_folder, f"{base}_QA.pdf")

    rows = []
    for q in ques_list:
        print("Question:", q)
        ans = answer_generation_chain.run(q)  # RetrievalQA supports .run for string input
        ans = " ".join(str(ans).split())
        print("Answer:", ans)
        print("-" * 50)
        rows.append([q, ans])

    _write_csv(csv_path, rows)
    _write_pdf(pdf_path, rows)

    return csv_path, pdf_path

@app.post("/analyze")
async def analyze(
    pdf_filename: str = Form(...),
    model_name: str = Form("llama-3.3-70b-versatile"),
):
    try:
        if not os.path.exists(pdf_filename):
            raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_filename}. Upload first.")

        csv_path, pdf_path = await asyncio.to_thread(generate_outputs_sync, pdf_filename, model_name)
        return JSONResponse({"output_file_csv": csv_path, "output_file_pdf": pdf_path})
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("ANALYZE_ERROR:\n", tb)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb[:4000]})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
