import io
import os
import tempfile
from pathlib import Path

import pyembroidery
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI(
    title="Embroidery Format Converter API",
    description="Convert between 90+ embroidery formats using pyembroidery.",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://embroiderydigitize.com",
    "https://www.embroiderydigitize.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported formats from pyembroidery — expand all alternative extensions
def _build_formats():
    seen = set()
    result = []
    for fmt in pyembroidery.supported_formats():
        if not isinstance(fmt, dict):
            continue
        description = fmt.get("description", "")
        readable = fmt.get("reader") is not None
        writable = fmt.get("writer") is not None
        # collect primary + all alternative extensions
        extensions = list(fmt.get("extensions") or [])
        primary = fmt.get("extension", "")
        if primary and primary not in extensions:
            extensions.insert(0, primary)
        for ext in extensions:
            ext = ext.lstrip(".").lower()
            if not ext or ext in seen:
                continue
            seen.add(ext)
            result.append({
                "extension": ext,
                "description": description,
                "read": readable,
                "write": writable,
            })
    return result

SUPPORTED_FORMATS = _build_formats()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/formats", summary="List all supported formats")
def get_formats():
    """Returns all formats supported for reading and/or writing."""
    return {
        "count": len(SUPPORTED_FORMATS),
        "formats": SUPPORTED_FORMATS,
    }


@app.post("/info", summary="Get embroidery file metadata")
async def get_info(file: UploadFile = File(...)):
    """
    Upload an embroidery file and receive metadata:
    stitch count, thread colors, and design dimensions.
    """
    suffix = Path(file.filename).suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="File has no extension.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pattern = pyembroidery.read(tmp_path)
        if pattern is None:
            raise HTTPException(status_code=422, detail="Could not read embroidery file. Unsupported or corrupt format.")

        stitch_count = len(pattern.stitches)
        thread_count = len(pattern.threadlist)

        # Bounding box
        min_x = min((s[0] for s in pattern.stitches), default=0)
        max_x = max((s[0] for s in pattern.stitches), default=0)
        min_y = min((s[1] for s in pattern.stitches), default=0)
        max_y = max((s[1] for s in pattern.stitches), default=0)

        # pyembroidery uses 0.1mm units
        width_mm = round((max_x - min_x) / 10, 2)
        height_mm = round((max_y - min_y) / 10, 2)

        colors = [
            {"name": t.name, "color": f"#{t.color:06X}" if t.color is not None else None}
            for t in pattern.threadlist
        ]

        return {
            "filename": file.filename,
            "stitch_count": stitch_count,
            "thread_count": thread_count,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "colors": colors,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/convert", summary="Convert embroidery file to another format")
async def convert(
    file: UploadFile = File(...),
    target_format: str = Query(..., description="Target format extension, e.g. dst, pes, jef"),
):
    """
    Upload an embroidery file and convert it to the specified target format.
    Returns the converted file as a downloadable response.
    """
    source_suffix = Path(file.filename).suffix.lower()
    target_ext = target_format.lstrip(".").lower()

    if not source_suffix:
        raise HTTPException(status_code=400, detail="Uploaded file has no extension.")

    # Validate target format is writable
    writable = [f["extension"] for f in SUPPORTED_FORMATS if f["write"]]
    if target_ext not in writable:
        raise HTTPException(
            status_code=400,
            detail=f"Target format '.{target_ext}' is not supported for writing. Supported: {', '.join(sorted(writable))}",
        )

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=source_suffix) as src_tmp:
        src_tmp.write(contents)
        src_path = src_tmp.name

    out_path = src_path + f".{target_ext}"

    try:
        pattern = pyembroidery.read(src_path)
        if pattern is None:
            raise HTTPException(status_code=422, detail="Could not read embroidery file. Unsupported or corrupt format.")

        pyembroidery.write(pattern, out_path)

        if not os.path.exists(out_path):
            raise HTTPException(status_code=500, detail="Conversion failed: output file was not created.")

        with open(out_path, "rb") as f:
            data = f.read()

        stem = Path(file.filename).stem
        download_name = f"{stem}.{target_ext}"

        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )
    finally:
        os.unlink(src_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
