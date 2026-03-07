#!/bin/bash
# Clean run for all supported documents in data/

set -e
export REFINERY_RUNTIME_RULES_FILE="rubric/extraction_rules.yaml"
echo "Using rules file: $REFINERY_RUNTIME_RULES_FILE"

# Remove previous outputs
echo "Removing .refinery directory..."
rm -rf .refinery

# Process all supported files in data/
for f in data/*.{pdf,docx,md,png,jpg,jpeg,xlsx}; do
    if [ -f "$f" ]; then
        echo "Processing $f ..."
        # Stage 1: Triage (profile generation)
        python -m src.refinery.cli ingest "$f"
        # Stage 2: Structure Extraction (included in ingest)
        # Stage 3: Semantic Chunking (included in ingest)
        # Stage 4: PageIndex build
        python -m src.refinery.cli build-index "$f"
        # Stage 5: Query Interface Agent (example query)
        python -m src.refinery.cli query-interface --doc "$f" "What is the document about?"
    fi
done

echo "All documents processed."
