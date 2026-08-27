"""
GraphMatcher V1.0 (Neo4j Cypher Compiler & Alignment)
Chịu trách nhiệm dịch CommonSemanticIR (EntityNode, EventEdge) thành truy vấn Cypher
và thực thi Graph Alignment với Neo4j Database.
"""

import logging
from typing import Dict, List, Any, Optional
from core.semantic_ir import CommonSemanticIR, EntityNode, EventEdge, TemporalEdge

logger = logging.getLogger(__name__)

class GraphMatcher:
    def __init__(self, neo4j_driver=None):
        """
        Khởi tạo GraphMatcher.
        neo4j_driver: Kết nối tới Neo4j (như neo4j.GraphDatabase.driver)
        """
        self.driver = neo4j_driver

    def build_cypher_query(self, ir: CommonSemanticIR, video_candidates: List[str] = None) -> str:
        """
        Dịch QueryGraph (từ SemanticIR) thành câu truy vấn Cypher chuẩn.
        Mô hình Đồ thị dự kiến trên Neo4j:
        (v:Video) -[:HAS_FRAME]-> (f:Frame) -[:CONTAINS]-> (e:Entity {type, color, count})
        (e1) -[:ACTION {name: 'walk'}]-> (e2)
        (f1) -[:NEXT_FRAME]-> (f2)
        """
        if not ir.entities and not ir.event_edges:
            return ""

        match_clauses = []
        where_clauses = []
        
        # 1. Khởi tạo ràng buộc danh sách video (từ Qdrant Recall)
        if video_candidates:
            vid_str = ", ".join([f"'{v}'" for v in video_candidates])
            match_clauses.append(f"MATCH (v:Video) WHERE v.id IN [{vid_str}]")
            match_clauses.append("MATCH (v)-[:HAS_FRAME]->(f:Frame)")
        else:
            match_clauses.append("MATCH (f:Frame)<-[:HAS_FRAME]-(v:Video)")

        # 2. Xây dựng Entity MATCH
        for entity in ir.entities:
            var_name = entity.entity_id.replace("-", "_")
            e_type = entity.entity_type
            match_clauses.append(f"MATCH (f)-[:CONTAINS]->({var_name}:Entity {{type: '{e_type}'}})")
            
            # Xử lý attributes
            for k, v in entity.attributes.items():
                if isinstance(v, bool):
                    where_clauses.append(f"{var_name}.{k} = {str(v).lower()}")
                elif isinstance(v, str):
                    where_clauses.append(f"toLower({var_name}.{k}) = '{v.lower()}'")
                elif isinstance(v, (int, float)):
                    where_clauses.append(f"{var_name}.{k} = {v}")

        # 3. Xây dựng Event MATCH (Relations between entities)
        for event in ir.event_edges:
            e_var = event.event_id.replace("-", "_")
            subj = event.subject_id.replace("-", "_") if event.subject_id else None
            obj = event.object_id.replace("-", "_") if event.object_id else None
            action = event.action.lower()

            if subj and obj:
                match_clauses.append(f"MATCH ({subj})-[{e_var}:PERFORMS {{action: '{action}'}}]->({obj})")
            elif subj:
                match_clauses.append(f"MATCH ({subj})-[{e_var}:PERFORMS {{action: '{action}'}}]->()")

        # 4. Xây dựng Temporal Edges (Dành cho TRAKE)
        # Sắp xếp các event theo thời gian thực hiện để tìm chuỗi sự kiện chính xác
        for t_edge in ir.temporal_edges:
            from_ev = t_edge.from_event.replace("-", "_")
            to_ev = t_edge.to_event.replace("-", "_")
            where_clauses.append(f"toInteger(split({from_ev}.id, '_')[2]) < toInteger(split({to_ev}.id, '_')[2])")

        cypher = "\n".join(match_clauses)
        if where_clauses:
            cypher += "\nWHERE " + " AND ".join(where_clauses)
        
        # Sửa lại RETURN để trả về đúng danh sách các timestamp ứng với các Event theo đúng trình tự (Dành cho TRAKE)
        event_vars = [e.event_id.replace("-", "_") for e in ir.event_edges]
        if event_vars:
            timestamp_ret = ", ".join([f"toInteger(split({ev}.id, '_')[2])" for ev in event_vars])
            cypher += f"\nRETURN v.id AS video_id, count(f) AS frame_match_count, [{timestamp_ret}] AS matched_timestamps"
        else:
            cypher += "\nRETURN v.id AS video_id, count(f) AS frame_match_count, collect(f.timestamp) AS matched_timestamps"
            
        cypher += "\nORDER BY frame_match_count DESC LIMIT 50"

        return cypher

    def execute_alignment(self, ir: CommonSemanticIR, qdrant_candidates: List[str]) -> Dict[str, Any]:
        """
        Thực thi Cypher query trên Neo4j.
        Trả về dictionary Map[video_id -> GraphScore details]
        """
        if not self.driver or (not ir.entities and not ir.event_edges):
            # Fallback nếu chưa có Neo4j driver hoặc query không có đồ thị
            return {vid: {"graph_score": 0.0, "matched_entities": 0} for vid in qdrant_candidates}

        cypher_query = self.build_cypher_query(ir, qdrant_candidates)
        logger.info(f"Generated Cypher Query:\n{cypher_query}")

        results_map = {}
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                for record in result:
                    vid = record["video_id"]
                    frame_count = record["frame_match_count"]
                    # Base graph score dựa trên số lượng frame khớp logic đồ thị
                    results_map[vid] = {
                        "graph_score": min(1.0, frame_count * 0.2), 
                        "matched_timestamps": record["matched_timestamps"]
                    }
        except Exception as e:
            logger.error(f"Neo4j Execution Error: {e}")
            return {vid: {"graph_score": 0.0, "error": str(e)} for vid in qdrant_candidates}

        # Bổ sung các candidate Qdrant không tìm thấy trong Graph với score 0
        for vid in qdrant_candidates:
            if vid not in results_map:
                results_map[vid] = {"graph_score": 0.0}

        return results_map
