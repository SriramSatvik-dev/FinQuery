from langgraph.graph import StateGraph, START, END
from src.graph.state import RAGState
from src.graph.nodes import query_classifier, hyde_query_gen, retrieve, reranker, abstention, write_ans, grounding_check, draft_response
from src.graph.edges import route_by_query_type, route_by_confidence, route_after_grounding

graph = StateGraph(RAGState)

graph.add_node(query_classifier)
graph.add_node(hyde_query_gen)
graph.add_node(retrieve)
graph.add_node(reranker)
graph.add_node(abstention)
graph.add_node(write_ans)
graph.add_node(grounding_check)
graph.add_node(draft_response)

graph.add_edge(START, 'query_classifier')
graph.add_conditional_edges('query_classifier', route_by_query_type)
graph.add_edge('hyde_query_gen', 'retrieve')
graph.add_edge('retrieve', 'reranker')
graph.add_conditional_edges('reranker', route_by_confidence)
graph.add_edge('abstention', END)
graph.add_edge('write_ans', 'grounding_check')
graph.add_conditional_edges('grounding_check', route_after_grounding)
graph.add_edge('draft_response', END)

pipeline = graph.compile()

def run_pipeline(query_str: str) -> dict :
    result = pipeline.invoke({'query': query_str})
    return result['final_response']