"""
import_csv_to_neo4j.py
Đọc các file CSV Event Graph (bản đã gộp hoặc từng part) và import siêu tốc vào Neo4j bằng UNWIND Batch.
"""
import os
import glob
import pandas as pd
from neo4j import GraphDatabase
import argparse
from tqdm import tqdm


def import_csv_to_neo4j(csv_path: str, uri: str, user: str, password: str, batch_size: int = 1000):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # 1. Tìm các file CSV cần nạp
    csv_files = []
    if os.path.isfile(csv_path):
        csv_files = [csv_path]
    elif os.path.isdir(csv_path):
        csv_files = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
    
    if not csv_files:
        print(f"⚠️ Không tìm thấy file CSV nào tại: {csv_path}")
        return

    print(f"🚀 Kết nối Neo4j tại {uri}...")
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (v:Video) ON (v.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (f:Frame) ON (f.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (a:ActionNode) ON (a.name)")
    
    batch_query = """
    UNWIND $batch AS row
    MERGE (v:Video {id: row.video_id})
    MERGE (f:Frame {id: row.frame_id})
    ON CREATE SET f.timestamp = row.timestamp, f.frame_idx = row.frame_idx
    MERGE (v)-[:HAS_FRAME]->(f)
    MERGE (e:Entity {id: row.entity_node_id})
    ON CREATE SET e.type = row.entity_type
    MERGE (f)-[:CONTAINS]->(e)
    MERGE (dummy:ActionNode {name: row.action})
    MERGE (e)-[rel:PERFORMS {action: row.action}]->(dummy)
    ON CREATE SET rel.id = row.rel_id
    """

    total_events = 0
    for file_path in csv_files:
        print(f"\n📦 Đang xử lý: {file_path}")
        df = pd.read_csv(file_path)
        if df.empty or "video_id" not in df.columns:
            continue

        # Chuẩn hóa dữ liệu theo batch
        records = []
        for _, row in df.iterrows():
            video_id = str(row['video_id'])
            try:
                timestamp = float(row['timestamp'])
            except (ValueError, TypeError):
                continue
            frame_idx = int(timestamp * 25)
            entity_id_raw = str(row.get('entity_id', '0'))
            entity_type = str(row.get('entity_type', 'object')).lower().strip()
            action = str(row.get('action', 'unknown')).lower().strip()

            frame_id = f"{video_id}_{frame_idx}"
            entity_node_id = f"{video_id}_{frame_idx}_{entity_id_raw}"
            rel_id = f"{video_id}_{frame_idx}_EV"

            records.append({
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "entity_node_id": entity_node_id,
                "entity_type": entity_type,
                "action": action,
                "rel_id": rel_id
            })

        print(f"⚡ Đang nạp {len(records):,} dòng vào Neo4j (Batch size = {batch_size})...")
        with driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Importing"):
                chunk = records[i:i + batch_size]
                session.run(batch_query, batch=chunk)
                total_events += len(chunk)

    print(f"\n✅ Đã import thành công tổng cộng {total_events:,} sự kiện vào Neo4j!")
    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/event_graph_merged_clean_395.csv", help="Đường dẫn file CSV hoặc thư mục CSV")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j Username")
    parser.add_argument("--password", type=str, default="password", help="Neo4j Password")
    parser.add_argument("--batch_size", type=int, default=2000, help="Kích thước batch khi nạp Neo4j")
    args = parser.parse_args()
    
    import_csv_to_neo4j(args.csv_path, args.uri, args.user, args.password, args.batch_size)
