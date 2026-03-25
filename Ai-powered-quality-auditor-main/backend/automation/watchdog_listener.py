import os
import time
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from watchdog.events import FileSystemEventHandler

from backend.core.multilingual_transcribe import get_transcription_engine
from backend.core.pii_masking import get_masking_pipeline
from backend.auditor_service import EnterpriseQualityAuditorService

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {'.mp3', '.wav', '.ogg', '.flac'}

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, incoming_dir: str, processed_dir: str, db_file: str):
        self.incoming_dir = Path(incoming_dir)
        self.processed_dir = Path(processed_dir)
        self.db_file = Path(db_file)
        
        # Initialize services
        self.transcription_engine = get_transcription_engine()
        self.masking_pipeline = get_masking_pipeline()
        self.auditor_service = EnterpriseQualityAuditorService()
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            return
            
        logger.info(f"New audio file detected: {file_path.name}")
        
        # Wait a moment to ensure file is fully written
        time.sleep(2)
        
        try:
            self.process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    def process_file(self, file_path: Path):
        filename = file_path.name
        
        logger.info(f"[{filename}] Step 1 - Transcribing audio...")
        transcription_result = self.transcription_engine.transcribe_and_process(str(file_path))
        transcript = transcription_result.transcript
        
        logger.info(f"[{filename}] Step 2 - Applying PII masking...")
        masking_result = self.masking_pipeline.process_transcript(filename, transcript)
        masked_transcript = masking_result.masked_text
        pii_count = masking_result.pii_count
        
        logger.info(f"[{filename}] Step 3 - Running quality audit...")
        audit_result = self.auditor_service.audit_transcript(masked_transcript)
        
        quality = audit_result.get("quality_score", {})
        
        # Attempt to map heuristic/LLM quality scores
        empathy = int(quality.get("empathy", quality.get("empathy_score", 0)))
        professionalism = int(quality.get("professionalism", quality.get("professionalism_score", 0)))
        resolution = int(quality.get("resolution", quality.get("overall_score", 0)))
        compliance = int(quality.get("compliance", quality.get("compliance_score", quality.get("compliance_status", 0))))
        
        # Sometimes compliance comes as a string ("Pass"/"Fail"), let's ensure it's an int for the dashboard
        if isinstance(compliance, str):
            compliance = 100 if compliance.lower() == "pass" else 0
        
        scores = {
            "empathy": empathy,
            "professionalism": professionalism,
            "resolution": resolution,
            "compliance": compliance
        }
        
        record = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "scores": scores,
            "pii_detected": pii_count
        }
        
        logger.info(f"[{filename}] Step 4 - Saving results...")
        self.save_to_db(record)
        
        logger.info(f"[{filename}] Moving to processed folder...")
        shutil.move(str(file_path), str(self.processed_dir / filename))
        logger.info(f"[{filename}] Processing complete.")
        
    def save_to_db(self, record):
        records = []
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        records = json.loads(content)
            except Exception as e:
                logger.error(f"Error reading DB file: {e}")
                
        records.append(record)
        
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to DB file: {e}")
