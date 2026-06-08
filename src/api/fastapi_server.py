from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import redis
import json
import logging
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

app = FastAPI(title="MedSeg-LLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

segmenter = None
report_generator = None


class SegmentationResponse(BaseModel):
    scan_id: str
    num_segments: int
    masks: List[Dict]
    iou_score: Optional[float] = None
    processing_time_ms: float


class ReportResponse(BaseModel):
    scan_id: str
    report: Dict
    processing_time_ms: float


@app.on_event("startup")
async def startup_event():
    global segmenter, report_generator
    logger.info("Loading MedSeg-LLM models...")
    # Model loading
    logger.info("Models loaded")


@app.post("/segment", response_model=SegmentationResponse)
async def segment_ct_scan(
    file: UploadFile = File(...),
    scan_id: Optional[str] = None,
    confidence_threshold: float = 0.5
):
    start_time = datetime.now()
    
    if not scan_id:
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cache_key = f"segmentation:{scan_id}"
    cached = redis_client.get(cache_key)
    
    if cached:
        result = json.loads(cached)
        result['cached'] = True
        return SegmentationResponse(**result)
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        if segmenter:
            results = segmenter.segment(image_array, confidence_threshold)
            masks = results['masks']
            scores = results['scores']
            boxes = results['boxes']
        else:
            masks = []
            scores = []
            boxes = []
        
        mask_data = [
            {
                'mask_id': i,
                'confidence': float(score),
                'box': box.tolist() if isinstance(box, np.ndarray) else box
            }
            for i, (mask, score, box) in enumerate(zip(masks, scores, boxes))
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            'scan_id': scan_id,
            'num_segments': len(masks),
            'masks': mask_data,
            'processing_time_ms': processing_time
        }
        
        redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return SegmentationResponse(**result)
    
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(scan_id: str, findings: List[Dict]):
    start_time = datetime.now()
    
    cache_key = f"report:{scan_id}"
    cached = redis_client.get(cache_key)
    
    if cached:
        result = json.loads(cached)
        result['cached'] = True
        return ReportResponse(**result)
    
    try:
        if report_generator:
            report = report_generator.generate_structured_report(findings)
        else:
            report = {
                'findings': 'Report generation not available',
                'impression': 'Model not loaded'
            }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            'scan_id': scan_id,
            'report': report,
            'processing_time_ms': processing_time
        }
        
        redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return ReportResponse(**result)
    
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'segmenter_loaded': segmenter is not None,
        'report_generator_loaded': report_generator is not None,
        'redis_connected': redis_client.ping()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
