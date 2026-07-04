from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))