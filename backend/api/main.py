"""
FastAPI Application Entry Point
Main REST and WebSocket API for the Quality Auditor system.

Architecture:
- REST endpoints for batch operations
- WebSocket endpoint for real-time streaming
- Proper middleware and error handling
- CORS enabled for frontend
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import audit service and supporting modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.auditor_service import EnterpriseQualityAuditorService
from backend.core.pii_masking import get_masking_pipeline
from backend.core.multilingual_transcribe import get_transcription_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== LIFECYCLE MANAGEMENT ====================

# Global instances
audit_service: Optional[EnterpriseQualityAuditorService] = None
masking_pipeline = get_masking_pipeline(enable_ner=True)
transcription_engine = get_transcription_engine(whisper_model="tiny", auto_translate=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    Initialize services and cleanup resources.
    """
    # Startup
    global audit_service
    logger.info("🚀 Starting AI Quality Auditor Service...")
    
    try:
        audit_service = EnterpriseQualityAuditorService(config={
            "enable_llm": True,
            "scoring_interval": 10.0,
            "anomaly_sensitivity": 2.0
        })
        logger.info("✅ Audit service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize audit service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Quality Auditor Service...")
    audit_service = None
    logger.info("✅ Service shutdown complete")


# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="AI Quality Auditor API",
    description="Production-ready AI-powered Customer Support Quality Auditor",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== REST ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AI Quality Auditor",
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "realtime_ws": "/ws/realtime",
            "batch_audit": "/audit/batch",
            "start_realtime": "/audit/realtime/start",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """System health check endpoint"""
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "service": "online",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - audit_service.service_start_time).total_seconds(),
        "total_conversations": audit_service.total_conversations,
        "total_segments": audit_service.total_segments
    }


# ==================== TRANSCRIPTION ENDPOINTS ====================

@app.post("/transcribe", tags=["Transcription"])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file with automatic language detection.
    
    Supports: MP3, WAV, OGG, FLAC
    Returns: Transcript with detected language and confidence
    """
    import tempfile
    import asyncio
    
    try:
        # Save uploaded file to OS-appropriate temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Audio file saved to {temp_path} ({len(content)} bytes)")
        
        # Run transcription in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(
            transcription_engine.transcribe_and_process, temp_path
        )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "data": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== PII MASKING ENDPOINTS ====================

@app.post("/pii/mask", tags=["PII Masking"])
async def mask_pii(request: Dict[str, str]):
    """
    Detect and mask PII in text.
    
    Request body:
    {
        "text": "John's SSN is 123-45-6789 and email is john@example.com"
    }
    
    Returns: Masked text with PII detection summary
    """
    try:
        text = request.get("text", "")
        if not text:
            raise ValueError("Text field is required")
        
        result = masking_pipeline.masker.mask(text)
        
        return {
            "success": True,
            "masked_text": result.masked_text,
            "pii_detected": len(result.detected_pii),
            "pii_summary": masking_pipeline.masker.get_pii_summary(result),
            "processing_time_ms": result.processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"PII masking error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== BATCH AUDIT ENDPOINT ====================

@app.post("/audit/batch", tags=["Audit"])
async def batch_audit(request: Dict[str, Any]):
    """
    Perform batch audit on a complete conversation.
    
    Request body:
    {
        "conversation_id": "conv_123",
        "agent_id": "agent_456",
        "transcript": "Full conversation text",
        "metadata": {}
    }
    
    Returns: Complete audit results with scores, violations, suggestions
    """
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        conversation_id = request.get("conversation_id", f"conv_{datetime.now().timestamp()}")
        agent_id = request.get("agent_id", "unknown")
        transcript = request.get("transcript", "")
        
        if not transcript:
            raise ValueError("Transcript is required")
        
        # Step 1: Mask PII
        masked_transcript = masking_pipeline.process_for_llm(transcript)
        
        # Step 2: Force-clean any existing conversation state
        if conversation_id in audit_service.streaming_engine.active_conversations:
            audit_service.streaming_engine.active_conversations.pop(conversation_id)
        # Clear stale audit results for this conversation
        audit_service.streaming_engine.audit_results = [
            r for r in audit_service.streaming_engine.audit_results
            if not r.segment_id.startswith(conversation_id)
        ]
        
        # Start fresh audit
        audit_service.start_realtime_audit(conversation_id, agent_id)
        
        # Step 3: Parse transcript into conversation turns
        lines = transcript.split('\n')
        agent_msg = ""
        customer_msg = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith("agent:"):
                # Process pending turn
                if agent_msg or customer_msg:
                    audit_service.process_realtime_segment(
                        conversation_id,
                        agent_message=agent_msg.strip(),
                        customer_message=customer_msg.strip(),
                        agent_id=agent_id
                    )
                    agent_msg = ""
                    customer_msg = ""
                # Start new agent msg
                # Remove prefix "Agent:" or "agent:"
                agent_msg = line[6:].strip()
            elif line.lower().startswith("customer:"):
                customer_msg = line[9:].strip()
            else:
                if customer_msg:
                    customer_msg += " " + line
                else:
                    agent_msg += " " + line
                    
        # Process the final turn
        if agent_msg or customer_msg:
            audit_service.process_realtime_segment(
                conversation_id,
                agent_message=agent_msg.strip(),
                customer_message=customer_msg.strip(),
                agent_id=agent_id
            )
        
        # Step 4: End audit and get results
        results = audit_service.end_realtime_audit(conversation_id, agent_id)
        
        # Step 5: Extract real scores — no artificial fallbacks or clamping
        final_report = results.get("final_report", {})
        metrics = final_report.get("metrics", {})
        
        empathy = int(metrics.get("empathy_avg", 0))
        professionalism = int(metrics.get("professionalism_avg", 0))
        resolution = int(metrics.get("resolution_avg", 0))
        compliance = int(metrics.get("compliance_avg", 0))
        
        # Extract escalation risk from the latest audit result
        escalation_risk = 0
        try:
            conv_results = [r for r in audit_service.streaming_engine.audit_results
                           if r.segment_id.startswith(conversation_id)]
            if conv_results:
                escalation_risk = conv_results[-1].quality_score.get("escalation_risk", 0)
        except Exception:
            pass
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "audit_results": {
                "final_report": {
                    "metrics": {
                        "empathy_avg": empathy,
                        "professionalism_avg": professionalism,
                        "overall_score": resolution,
                        "compliance_avg": compliance
                    }
                }
            },
            "scores": {
                "empathy": empathy,
                "professionalism": professionalism,
                "resolution": resolution,
                "compliance": compliance
            },
            "escalation_risk": escalation_risk,
            "pii_masked": True,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch audit error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== REAL-TIME STREAMING ENDPOINTS ====================

@app.post("/audit/realtime/start", tags=["RealTime Audit"])
async def start_realtime_audit(request: Dict[str, str]):
    """
    Start a new real-time audit session.
    
    Request body:
    {
        "conversation_id": "conv_123",
        "agent_id": "agent_456"
    }
    
    Returns: Session initialization confirmation
    """
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        conversation_id = request.get("conversation_id", f"conv_{datetime.now().timestamp()}")
        agent_id = request.get("agent_id", "unknown")
        
        result = audit_service.start_realtime_audit(conversation_id, agent_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "session_status": result,
            "ws_endpoint": f"/ws/realtime?conversation_id={conversation_id}",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to start realtime audit: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/realtime")
async def websocket_realtime_audit(websocket: WebSocket):
    """
    WebSocket endpoint for real-time conversation audit.
    
    Connection protocol:
    1. Client connects: /ws/realtime?conversation_id=xxx
    2. Client sends: {"agent": "...", "customer": "..."}
    3. Server responds: {"scores": {...}, "alerts": [...]}
    4. Client closes to end session
    
    Message format (incoming):
    {
        "agent": "Agent message",
        "customer": "Customer message"
    }
    
    Message format (outgoing):
    {
        "scores": {"empathy": 85, "professionalism": 90},
        "compliance": "PASS",
        "alerts": ["Alert 1"],
        "timestamp": "2026-03-03T..."
    }
    """
    if audit_service is None:
        await websocket.close(code=1008, reason="Service not initialized")
        return
    
    try:
        await websocket.accept()
        
        # Extract conversation_id from query params
        conversation_id = websocket.query_params.get("conversation_id", f"ws_{datetime.now().timestamp()}")
        agent_id = websocket.query_params.get("agent_id", "unknown")
        
        logger.info(f"WebSocket connected: {conversation_id}")
        
        # Start audit session
        await websocket.send_json({
            "type": "session_started",
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process streaming messages
        while True:
            data = await websocket.receive_json()
            
            agent_message = data.get("agent", "")
            customer_message = data.get("customer", "")
            
            if not agent_message and not customer_message:
                continue
            
            # Mask PII before processing
            agent_message = masking_pipeline.process_for_llm(agent_message)
            customer_message = masking_pipeline.process_for_llm(customer_message)
            
            # Process segment
            result = audit_service.process_realtime_segment(
                conversation_id,
                agent_message=agent_message,
                customer_message=customer_message,
                agent_id=agent_id
            )
            
            # Send back results
            await websocket.send_json({
                "type": "segment_processed",
                "scores": result.get("scores", {}),
                "compliance": result.get("compliance", "UNKNOWN"),
                "alerts": result.get("alerts", []),
                "suggestions": result.get("suggestions", []),
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    finally:
        logger.info(f"WebSocket disconnected: {conversation_id}")


@app.post("/audit/realtime/end", tags=["RealTime Audit"])
async def end_realtime_audit(request: Dict[str, str]):
    """
    End a real-time audit session and generate final report.
    
    Request body:
    {
        "conversation_id": "conv_123",
        "agent_id": "agent_456"
    }
    
    Returns: Final audit report with coaching insights
    """
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        conversation_id = request.get("conversation_id", "")
        agent_id = request.get("agent_id", "unknown")
        
        if not conversation_id:
            raise ValueError("conversation_id is required")
        
        result = audit_service.end_realtime_audit(conversation_id, agent_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "final_report": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to end realtime audit: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standard format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== MAIN ====================

if __name__ == "__main__":
    # Run server
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )
