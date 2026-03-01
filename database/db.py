


from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class CallRecord(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    transcript = Column(Text)
    score = Column(String)
    feedback = Column(Text)

Base.metadata.create_all(bind=engine)
