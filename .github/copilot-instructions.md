# Embroidery Format Converter API

## Stack
- **FastAPI** + **uvicorn** (dev server with --reload)
- **pyembroidery** — reads/writes 90+ embroidery formats
- **python-multipart** — file upload support

## CORS Origins
- http://localhost:3000 (Next.js local dev)
- https://embroiderydigitize.com
- https://www.embroiderydigitize.com

## Endpoints
- `GET  /formats`  — list all supported formats (read/write flags)
- `POST /info`     — upload file → returns stitch count, colors, dimensions
- `POST /convert`  — upload file + `?target_format=dst` → returns converted file

## Running Locally
```
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Interactive docs: http://127.0.0.1:8000/docs

## Project Setup Checklist

- [x] Create copilot-instructions.md
- [x] Clarify Project Requirements
- [x] Scaffold the Project
- [x] Customize the Project
- [x] Install Required Extensions
- [x] Compile the Project
- [x] Create and Run Task
- [x] Launch the Project
- [x] Ensure Documentation is Complete
