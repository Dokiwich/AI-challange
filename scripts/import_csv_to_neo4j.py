"""
import_csv_to_neo4j.py
Đọc các file event_graph_part*.csv sinh ra từ Colab và import vào Neo4j Graph Database.
"""
import os
import glob
import pandas as pd
from neo4j import GraphDatabase
import argparse
from tqdm import tqdm

def import_csv_to_neo4j(csv_dir: str, uri: str, user: str, password: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Tìm tất cả file csv
    csv_files = glob.glob(os.path.join(csv_dir, "event_graph_part*.csv"))
    if not csv_files:
        print(f"⚠️ Không tìm thấy file event_graph_part*.csv nào trong {csv_dir}")
        return

    print("🚀 Khởi tạo Indexes và Constraints...")
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (v:Video) ON (v.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (f:Frame) ON (f.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
    
    total_events = 0
    for csv_file in csv_files:
        print(f"\n📦 Đang xử lý file: {csv_file}")
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
            
        with driver.session() as session:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                video_id = str(row['video_id'])
                timestamp = float(row['timestamp'])
                # Tính frame = timestamp * 25 (giả định fps=25)
                frame_idx = int(timestamp * 25) 
                entity_id_raw = str(row['entity_id'])
                entity_type = str(row['entity_type']).lower()
                action = str(row['action']).lower()
                
                # Tạo node Frame và Entity có id duy nhất theo thời gian
                frame_id = f"{video_id}_{frame_idx}"
                entity_node_id = f"{video_id}_{frame_idx}_{entity_id_raw}"
                rel_id = f"{video_id}_{frame_idx}_EV"
                
                query = """
                // 1. Tạo hoặc lấy Video Node
                MERGE (v:Video {id: $video_id})
                
                // 2. Tạo Frame Node
                MERGE (f:Frame {id: $frame_id})
                ON CREATE SET f.timestamp = $timestamp, f.frame_idx = $frame_idx
                
                // 3. Liên kết Video -> Frame
                MERGE (v)-[:HAS_FRAME]->(f)
                
                // 4. Tạo Entity Node
                MERGE (e:Entity {id: $entity_node_id})
                ON CREATE SET e.type = $entity_type
                
                // 5. Liên kết Frame -> Entity
                MERGE (f)-[:CONTAINS]->(e)
                
                // 6. Tạo Action Relationship (Gắn ID để graph_matcher có thể so sánh thời gian)
                MERGE (e)-[rel:PERFORMS {action: $action}]->(dummy:ActionNode {name: $action})
                ON CREATE SET rel.id = $rel_id
                """
                session.run(query, 
                            video_id=video_id, 
                            frame_id=frame_id, 
                            timestamp=timestamp, 
                            frame_idx=frame_idx,
                            entity_node_id=entity_node_id,
                            entity_type=entity_type,
                            action=action,
                            rel_id=rel_id)
                total_events += 1
                
    print(f"\n✅ Đã import thành công {total_events} sự kiện vào Neo4j!")
    driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default=".", help="Thư mục chứa file csv")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j Username")
    parser.add_argument("--password", type=str, default="password", help="Neo4j Password")
    args = parser.parse_args()
    
    import_csv_to_neo4j(args.csv_dir, args.uri, args.user, args.password)
