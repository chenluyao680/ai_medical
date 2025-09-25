from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration import config


class IndexUtil:
    def __init__(self):
        self.graph = Neo4jGraph(url=config.NEO4J_CONFIG["uri"],
                                username=config.NEO4J_CONFIG['auth'][0],
                                password=config.NEO4J_CONFIG['auth'][1])

        self.embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5',
                                                     encode_kwargs={'normalize_embeddings': True})

    def create_full_text_index(self, index_name, label, property):
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON EACH [n.{property}]
        """
        self.graph.query(cypher)

    def create_vector_index(self, index_name, label, source_property, embedding_property):
        embedding_dim = self._add_embedding(label, source_property, embedding_property)

        cypher = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (m:{label})
                ON m.{embedding_property}
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {embedding_dim},
                  `vector.similarity_function`: 'cosine'
                    }}
                }}
        """
        self.graph.query(cypher)

    def _add_embedding(self, label, source_property, embedding_property):
        cypher = f"""
            MATCH (n:{label}) RETURN n.{source_property} AS text,id(n) AS id
        """
        results = self.graph.query(cypher)
        docs = [result['text'] for result in results]
        # 过滤并处理文本
        valid_docs = []
        for doc in docs:
            if doc is None:
                valid_docs.append("")  # 替换 None 为空字符串
            elif not isinstance(doc, str):
                valid_docs.append(str(doc))  # 转换为字符串
            else:
                valid_docs.append(doc)

        # 处理换行符
        processed_docs = [doc.replace("\n", " ") for doc in valid_docs]

        # 生成嵌入
        embeddings = self.embedding_model.embed_documents(processed_docs)

        batch = []
        for result, embedding in zip(results, embeddings):
            item = {
                'id': result['id'],
                'embedding': embedding
            }
            batch.append(item)

        cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{label}) WHERE id(n) = item.id
            SET n.{embedding_property} = item.embedding
        """
        self.graph.query(cypher, params={'batch': batch})
        return len(embeddings[0])


if __name__ == '__main__':
    # 创建索引
    index_util = IndexUtil()