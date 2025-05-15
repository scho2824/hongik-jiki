#!/usr/bin/env python3
"""
Run Tagging Pipeline for Hongik-Jiki Chatbot

This script processes documents and assigns tags based on content analysis.
"""
import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Add project root to path
sys.path.append(str(ROOT_DIR))

from hongikjiki.modules.tagging.tag_schema import TagSchema
from hongikjiki.modules.tagging.tag_extractor import TagExtractor
from hongikjiki.modules.tagging.tag_analyzer import TagAnalyzer
from hongikjiki.modules.tagging.tagging_tools import TaggingBatch, TagValidationTool
from hongikjiki.modules.vector_store.tag_index import TagIndex


def setup_logging(log_file: str | Path = "logs/tagging.log") -> logging.Logger:
    log_file = Path(log_file) if isinstance(log_file, str) else log_file
    os.makedirs(log_file.parent, exist_ok=True)

    logger = logging.getLogger("HongikJikiTagger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_documents(input_dir: str) -> List[Dict[str, Any]]:
    documents = []

    input_path = Path(input_dir)
    if not input_path.exists():
        return documents

    for file_path in input_path.iterdir():
        if not file_path.name.endswith(".json"):
            continue

        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)

            doc = {
                "id": data.get("id") or file_path.stem,
                "content": data.get("content") or data.get("text", "") or data.get("page_content", ""),
                "metadata": data.get("metadata", {})
            }

            if doc["content"]:
                documents.append(doc)
        except Exception as e:
            logging.error(f"Error loading document {file_path.name}: {e}")

    return documents


def run_tagging_pipeline(args: argparse.Namespace) -> None:
    logger = setup_logging(args.log_file)
    logger.info("Starting tagging pipeline")

    logger.info("Initializing tag schema")
    tag_schema = TagSchema(args.tag_schema)

    logger.info("Initializing tag extractor")
    tag_extractor = TagExtractor(tag_schema, args.tag_patterns)

    logger.info("Initializing tag analyzer")
    tag_analyzer = TagAnalyzer(tag_schema)
    stats_file_path = Path(args.stats_file)
    if stats_file_path.exists():
        tag_analyzer.load_tag_statistics(str(stats_file_path))

    logger.info("Initializing tag index")
    tag_index = TagIndex(args.index_file)

    logger.info(f"Loading documents from {args.input_dir}")
    documents = load_documents(args.input_dir)
    logger.info(f"Loaded {len(documents)} documents")

    if not documents:
        logger.warning("No documents found, exiting")
        return

    logger.info("Processing documents with batch tagger")
    batch_tagger = TaggingBatch(tag_schema, tag_extractor, args.output_dir)
    stats = batch_tagger.process_batch(documents, args.min_confidence)

    batch_tagger.export_batch_results(args.batch_report)
    logger.info(f"Batch processing complete: {stats['successfully_tagged']} documents tagged")

    if args.validate:
        logger.info("Validating tags")
        validator = TagValidationTool(tag_schema, tag_analyzer)
        validation_results = validator.batch_validate_tags(args.output_dir, args.validation_report)
        validator.export_tag_validation_report(validation_results, args.validation_csv, format="csv")
        logger.info(f"Validation complete: {validation_results['documents_with_issues']} documents with issues")

    logger.info("Updating tag statistics")
    output_path = Path(args.output_dir)
    for doc in documents:
        doc_id = doc["id"]
        tags_file = output_path / f"{doc_id}_tags.json"

        if tags_file.exists():
            try:
                with tags_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)

                tags = data.get("tags", {})
                if tags:
                    tag_analyzer.process_document_tags(doc_id, tags)
            except Exception as e:
                logger.error(f"Error processing tags file {tags_file}: {e}")

    logger.info("Saving tag statistics")
    tag_analyzer.save_tag_statistics(args.stats_file)

    logger.info("Rebuilding tag index")
    count = tag_index.rebuild_from_tag_files(args.output_dir)
    logger.info(f"Tag index rebuilt with {count} documents")

    logger.info("Tagging pipeline complete")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tagging pipeline for Hongik-Jiki Chatbot")

    parser.add_argument(
        "--input-dir", type=str, default=str(ROOT_DIR / "data/tag_data/input_chunks"),
        help="Directory containing document chunks"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(ROOT_DIR / "data/tag_data/auto_tagged"),
        help="Directory for saving tagged documents"
    )
    parser.add_argument(
        "--tag-schema", type=str, default=str(ROOT_DIR / "data/config/tag_schema.yaml"),
        help="Path to tag schema YAML file"
    )
    parser.add_argument(
        "--tag-patterns", type=str, default=str(ROOT_DIR / "data/config/tag_patterns.json"),
        help="Path to tag patterns JSON file"
    )
    parser.add_argument(
        "--stats-file", type=str, default=str(ROOT_DIR / "data/tag_data/tag_statistics.json"),
        help="Path to tag statistics file"
    )
    parser.add_argument(
        "--index-file", type=str, default=str(ROOT_DIR / "data/tag_data/tag_index.json"),
        help="Path to tag index file"
    )
    parser.add_argument(
        "--batch-report", type=str, default=str(ROOT_DIR / "data/tag_data/batch_report.json"),
        help="Path to batch processing report"
    )
    parser.add_argument(
        "--validation-report", type=str, default=str(ROOT_DIR / "data/tag_data/validation_report.json"),
        help="Path to tag validation report"
    )
    parser.add_argument(
        "--validation-csv", type=str, default=str(ROOT_DIR / "data/tag_data/validation_report.csv"),
        help="Path to tag validation CSV report"
    )
    parser.add_argument(
        "--log-file", type=str, default=str(ROOT_DIR / "logs/tagging.log"),
        help="Path to log file"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.6,
        help="Minimum confidence threshold for tags"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate tags after processing"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_tagging_pipeline(args)