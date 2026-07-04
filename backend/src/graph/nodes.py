from src.graph.state import RAGState
from src.classification.classify import classify_query
from src.classification.hyde_query import hyde_gen
from src.retrieval.bm25_retriever import query as bm25_query
from src.retrieval.dense_retriever import query as dense_query
from src.retrieval.merger import merge
from src.reranking.cross_encoder import rerank
from src.generation.generator import generate
from src.generation.grounding import check_grounding

def query_classifier(state: RAGState):
    return {
        'query_type': classify_query(state['query'])
    }

def hyde_query_gen(state: RAGState):
    passage = hyde_gen(state['query'])
    return {'hyde_passage': passage}

def retrieve(state: RAGState):
    bm25_chunks = bm25_query(state['query'])
    
    if state['query_type'] == 'conceptual':
        dense_chunks = dense_query(state.get('hyde_passage', state['query']))
    else:
        dense_chunks = dense_query(state['query'])
    
    return {'candidate_chunks': merge(bm25_chunks, dense_chunks)}

def reranker(state: RAGState):
    ranked_chunks, top_score = rerank(state['query'], state['candidate_chunks'])
    return {
        'reranked_chunks': ranked_chunks,
        'top_reranker_score': top_score
    }

def abstention(state: RAGState):
    return {
        'abstain': True,
        'final_response': {
            'answer': None,
            'abstained': True,
            'reason': "Insufficient corpus coverage",
            'message': "...",
            'citations': []
        }
    }

def write_ans(state: RAGState):
    answer, cited_chunks = generate(state['query'], state['reranked_chunks'])

    return {
        'answer': answer,
        'citations': cited_chunks
    }

def grounding_check(state: RAGState):
    passed = check_grounding(state['answer'], state['citations'], state['reranked_chunks'])

    if not passed:
        return {
            'grounding_passed': False,
            'abstain': True,
            'final_response': {
                'answer': None,
                'abstained': True,
                'reason': 'grounding_check_failed',
                'message': 'Generated answer could not be verified against source documents',
                'citations': []
            }
        }
    
    return {'grounding_passed': True}

def draft_response(state: RAGState):
    return {
        'final_response': {
            'answer': state['answer'],
            'abstained': False,
            'citations': [
                {
                    'chunk_id': c.chunk_id,
                    'source_file': c.source_file,
                    'page_num': c.page_num,
                    'excerpt': c.excerpt
                }
                for c in state['citations']
            ],
            'query_type': state['query_type'],
            'top_reranker_score': state['top_reranker_score']
        }
    }