from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

CONFIDENCE_THRESHOLD = os.getenv("CONFIDENCE_THRESHOLD")

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))