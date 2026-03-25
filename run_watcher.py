import os
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from backend.automation.watchdog_listener import AudioFileHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    incoming_dir = data_dir / "incoming_audio"
    processed_dir = data_dir / "processed_audio"
    db_file = data_dir / "analyzed_transcripts.json"
    
    # Create folders if missing
    for d in [data_dir, incoming_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    if not db_file.exists():
        with open(db_file, 'w') as f:
            f.write("[]")
            
    event_handler = AudioFileHandler(
        incoming_dir=str(incoming_dir),
        processed_dir=str(processed_dir),
        db_file=str(db_file)
    )
    
    observer = Observer()
    observer.schedule(event_handler, str(incoming_dir), recursive=False)
    observer.start()
    
    logger.info(f"Watching for new audio files in {incoming_dir}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
