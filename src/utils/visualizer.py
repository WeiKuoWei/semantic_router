import json
from pathlib import Path
from typing import Dict, Any


class CentroidVisualizer:
    def __init__(self, tracking_file: str, output_file: str):
        """
        Initialize the CentroidVisualizer.
        
        Args:
            tracking_file: Path to the JSON file containing centroid data
            output_file: Path to save the generated Python file
        """
        self.tracking_file = Path(tracking_file)
        self.output_file = Path(output_file)
    
    def generate_centroid_vectors_file(self):
        """
        Generate a Python file containing centroid vectors for the router.
        """
        if not self.tracking_file.exists():
            raise FileNotFoundError(f"Tracking file not found: {self.tracking_file}")
        
        # Load tracking data
        with open(self.tracking_file, 'r') as f:
            tracking_data = json.load(f)
        
        # Start building the output file content
        content = [
            "# Auto-generated centroid vectors file",
            "# This file contains centroid vectors for the multi-layer semantic router",
            "",
            "# Group-level centroids",
            "GROUP_CENTROIDS = {",
        ]
        
        # Add group centroids
        for group_name, group_data in tracking_data.items():
            if group_data["centroid"] is not None:
                centroid_str = str(group_data["centroid"])
                content.append(f'    "{group_name}": {centroid_str},')
        
        content.append("}")
        content.append("")
        content.append("# Expert-level centroids")
        content.append("EXPERT_CENTROIDS = {")
        
        # Add expert centroids
        for group_name, group_data in tracking_data.items():
            for expert_name, expert_data in group_data.get("experts", {}).items():
                if expert_data["centroid"] is not None:
                    centroid_str = str(expert_data["centroid"])
                    content.append(f'    "{expert_name}": {centroid_str},')
        
        content.append("}")
        content.append("")
        
        # Add expert to group mapping
        content.append("# Expert to group mapping")
        content.append("EXPERT_TO_GROUP = {")
        
        for group_name, group_data in tracking_data.items():
            for expert_name in group_data.get("experts", {}):
                content.append(f'    "{expert_name}": "{group_name}",')
        
        content.append("}")
        content.append("")
        
        # Add group to experts mapping
        content.append("# Group to experts mapping")
        content.append("GROUP_TO_EXPERTS = {")
        
        for group_name, group_data in tracking_data.items():
            expert_list = list(group_data.get("experts", {}).keys())
            content.append(f'    "{group_name}": {expert_list},')
        
        content.append("}")
        
        # Write the file
        with open(self.output_file, 'w') as f:
            f.write("\n".join(content))
        
        print(f"Generated centroid vectors file: {self.output_file}")