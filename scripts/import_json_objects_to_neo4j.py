import os
import glob
import json
import argparse
from tqdm import tqdm
from neo4j import GraphDatabase
from collections import defaultdict

def import_json_objects_to_neo4j(data_dir: str, map_file: str, uri: str, user: str, password: str, batch_size: int = 2000, threshold: float = 0.0):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # 1. Load keyframe mapping
    print(f"📦 Loading keyframe mapping from {map_file}...")
    frame_lookup = defaultdict(dict)
    if os.path.exists(map_file):
        with open(map_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            for item in mapping_data:
                video = item.get("video")
                kf_idx = item.get("keyframe_idx")
                f_idx = item.get("frame_idx")
                if video and kf_idx is not None and f_idx is not None:
                    frame_lookup[video][kf_idx] = f_idx
    else:
        print(f"⚠️ Warning: map_keyframes.json not found at {map_file}. Will fallback to frame_idx = keyframe_idx * 25.")

    print(f"🚀 Connecting to Neo4j at {uri}...")
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (v:Video) ON (v.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (f:Frame) ON (f.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
    
    # Query optimized for bulk merge
    batch_query = """
    UNWIND $batch AS row
    MERGE (v:Video {id: row.video_id})
    MERGE (f:Frame {id: row.frame_id})
    ON CREATE SET f.frame_idx = row.frame_idx
    MERGE (v)-[:HAS_FRAME]->(f)
    MERGE (e:Entity {id: row.entity_id})
    ON CREATE SET e.type = row.entity_type
    MERGE (f)-[:CONTAINS]->(e)
    """

    video_folders = sorted(glob.glob(os.path.join(data_dir, "*")))
    total_objects = 0

    for video_dir in tqdm(video_folders, desc="Processing Videos"):
        if not os.path.isdir(video_dir):
            continue
            
        video_id = os.path.basename(video_dir)
        json_files = sorted(glob.glob(os.path.join(video_dir, "*.json")))
        
        records = []
        for jf in json_files:
            try:
                kf_idx = int(os.path.splitext(os.path.basename(jf))[0])
            except ValueError:
                continue
                
            frame_idx = frame_lookup.get(video_id, {}).get(kf_idx)
            if frame_idx is None:
                # Fallback mapping if not in map_keyframes
                frame_idx = kf_idx * 25

            frame_id = f"{video_id}_{frame_idx}"

            with open(jf, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    continue
            
            entities = data.get("detection_class_entities", [])
            scores = data.get("detection_scores", [])
            
            # Ensure lists match length
            limit = min(len(entities), len(scores))
            
            type_counter = defaultdict(int)
            for i in range(limit):
                try:
                    score = float(scores[i])
                except:
                    score = 0.0
                
                # Filter by confidence threshold (user requested to load all, so threshold=0.0)
                if score >= threshold:
                    e_type = str(entities[i]).lower().strip()
                    type_counter[e_type] += 1
                    count_idx = type_counter[e_type]
                    
                    entity_id = f"{frame_id}_{e_type}_{count_idx}"
                    
                    records.append({
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "frame_id": frame_id,
                        "entity_id": entity_id,
                        "entity_type": e_type
                    })

        # Batch insert for the current video
        if records:
            with driver.session() as session:
                for i in range(0, len(records), batch_size):
                    chunk = records[i:i + batch_size]
                    session.run(batch_query, batch=chunk)
                    total_objects += len(chunk)

    print(f"\n✅ Đã import thành công {total_objects:,} objects vào Neo4j!")
    driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/objects", help="Thư mục chứa các folder video objects JSON")
    parser.add_argument("--map_file", type=str, default="data/mapping/map_keyframes.json", help="File JSON map keyframes")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j Username")
    parser.add_argument("--password", type=str, default="password", help="Neo4j Password")
    parser.add_argument("--batch_size", type=int, default=5000, help="Kích thước batch import")
    parser.add_argument("--threshold", type=float, default=0.0, help="Ngưỡng confidence score (0.0 = nạp hết)")
    
    args = parser.parse_args()
    import_json_objects_to_neo4j(
        data_dir=args.data_dir,
        map_file=args.map_file,
        uri=args.uri,
        user=args.user,
        password=args.password,
        batch_size=args.batch_size,
        threshold=args.threshold
    )
