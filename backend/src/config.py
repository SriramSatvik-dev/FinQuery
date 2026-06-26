from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))