from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
from typing import Optional

# Import functions from main.py
from main import (
    recommend_papers_tool,
    summarize_pdf_tool,
    extract_research_trends,
    generate_research_idea,
    evaluate_feasibility
)

# Create FastAPI app
app = FastAPI(title="Research Assistant API")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Mount templates and static files
templates = Jinja2Templates(directory="templates")

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend-papers", response_class=HTMLResponse)
async def recommend_papers(
    request: Request,
    query: str = Form(...),
    year: Optional[int] = Form(None),
    limit: int = Form(5)
):
    """API endpoint for recommending papers."""
    try:
        result = recommend_papers_tool(query, year, limit)
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "result": result, "action": "Paper Recommendations"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-pdf", response_class=HTMLResponse)
async def summarize_pdf(request: Request, url: str = Form(...)):
    """API endpoint for summarizing PDFs."""
    try:
        result = summarize_pdf_tool(url)
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "result": result, "action": "PDF Summary"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research-trends", response_class=HTMLResponse)
async def research_trends(
    request: Request,
    query: str = Form(...),
    year: int = Form(2025),
    limit: int = Form(10)
):
    """API endpoint for analyzing research trends."""
    try:
        result = extract_research_trends(query, year, limit)
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "result": result, "action": "Research Trends Analysis"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-idea", response_class=HTMLResponse)
async def generate_idea(
    request: Request,
    query: str = Form(...),
    year: int = Form(2025),
    limit: int = Form(5)
):
    """API endpoint for generating research ideas."""
    try:
        result = generate_research_idea(query, year, limit)
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "result": result, "action": "Research Idea Generation"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-feasibility", response_class=HTMLResponse)
async def evaluate_idea_feasibility(request: Request, research_idea: str = Form(...)):
    """API endpoint for evaluating research idea feasibility."""
    try:
        result = evaluate_feasibility(research_idea)
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "result": result, "action": "Feasibility Evaluation"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)