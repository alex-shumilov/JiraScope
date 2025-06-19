"""Export functionality for CLI results."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from rich.console import Console

console = Console()


def export_duplicate_report(report, output_path: str):
    """Export duplicate analysis report to file"""
    output_file = Path(output_path)
    
    if output_file.suffix.lower() == '.json':
        _export_duplicates_json(report, output_file)
    elif output_file.suffix.lower() == '.csv':
        _export_duplicates_csv(report, output_file)
    else:
        # Default to JSON
        output_file = output_file.with_suffix('.json')
        _export_duplicates_json(report, output_file)


def _export_duplicates_json(report, output_file: Path):
    """Export duplicates to JSON format"""
    export_data = {
        "summary": {
            "total_candidates": report.total_candidates,
            "processing_cost": report.processing_cost,
            "candidates_by_level": {
                level: len(candidates) for level, candidates in report.candidates_by_level.items()
            }
        },
        "candidates": []
    }
    
    # Flatten all candidates
    for level, candidates in report.candidates_by_level.items():
        for candidate in candidates:
            export_data["candidates"].append({
                "original_key": candidate.original_key,
                "duplicate_key": candidate.duplicate_key,
                "similarity_score": candidate.similarity_score,
                "confidence_level": candidate.confidence_level,
                "suggested_action": candidate.suggested_action
            })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)


def _export_duplicates_csv(report, output_file: Path):
    """Export duplicates to CSV format"""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Original Key', 'Duplicate Key', 'Similarity Score', 
            'Confidence Level', 'Suggested Action'
        ])
        
        for level, candidates in report.candidates_by_level.items():
            for candidate in candidates:
                writer.writerow([
                    candidate.original_key,
                    candidate.duplicate_key,
                    candidate.similarity_score,
                    candidate.confidence_level,
                    candidate.suggested_action
                ])


def export_quality_report(analyses: List, output_path: str):
    """Export quality analysis report to file"""
    output_file = Path(output_path)
    
    if output_file.suffix.lower() == '.json':
        _export_quality_json(analyses, output_file)
    elif output_file.suffix.lower() == '.csv':
        _export_quality_csv(analyses, output_file)
    else:
        # Default to JSON
        output_file = output_file.with_suffix('.json')
        _export_quality_json(analyses, output_file)


def _export_quality_json(analyses: List, output_file: Path):
    """Export quality analyses to JSON format"""
    export_data = {
        "summary": {
            "total_analyses": len(analyses),
            "average_score": sum(a.overall_score for a in analyses) / len(analyses) if analyses else 0,
            "total_cost": sum(a.analysis_cost for a in analyses)
        },
        "analyses": []
    }
    
    for analysis in analyses:
        export_data["analyses"].append({
            "work_item_key": analysis.work_item_key,
            "clarity_score": analysis.clarity_score,
            "completeness_score": analysis.completeness_score,
            "actionability_score": analysis.actionability_score,
            "testability_score": analysis.testability_score,
            "overall_score": analysis.overall_score,
            "risk_level": analysis.risk_level,
            "improvement_suggestions": analysis.improvement_suggestions,
            "analysis_cost": analysis.analysis_cost
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)


def _export_quality_csv(analyses: List, output_file: Path):
    """Export quality analyses to CSV format"""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Work Item Key', 'Clarity Score', 'Completeness Score',
            'Actionability Score', 'Testability Score', 'Overall Score',
            'Risk Level', 'Improvement Suggestions', 'Analysis Cost'
        ])
        
        for analysis in analyses:
            writer.writerow([
                analysis.work_item_key,
                analysis.clarity_score,
                analysis.completeness_score,
                analysis.actionability_score,
                analysis.testability_score,
                analysis.overall_score,
                analysis.risk_level,
                '; '.join(analysis.improvement_suggestions),
                analysis.analysis_cost
            ])


def export_generic_data(data: Union[Dict, List], output_path: str, title: str = "Export"):
    """Generic export function for any data structure"""
    output_file = Path(output_path)
    
    try:
        if output_file.suffix.lower() == '.json':
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # For non-JSON, try to convert to a simple format
            if isinstance(data, dict):
                with open(output_file, 'w') as f:
                    for key, value in data.items():
                        f.write(f"{key}: {value}\n")
            elif isinstance(data, list):
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(f"{item}\n")
        
        console.print(f"[green]✅ {title} exported to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Export failed: {e}[/red]")