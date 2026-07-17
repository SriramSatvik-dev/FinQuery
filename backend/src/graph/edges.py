from src.config import CONFIDENCE_THRESHOLD
from src.graph.state import RAGState
from langgraph.graph import END

def route_by_query_type(state: RAGState):
    if state['query_type'] == 'conceptual': return 'hyde_query_gen'
    return 'retrieve'

def route_by_confidence(state: RAGState):
    if state['top_reranker_score'] < CONFIDENCE_THRESHOLD: return 'abstention'
    return 'write_ans'

def route_after_grounding(state: RAGState):
    # if not state['grounding_passed']: return END
    return 'draft_response'