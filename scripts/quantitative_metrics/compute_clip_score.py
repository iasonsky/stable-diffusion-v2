import torch
import clip
from PIL import Image
import argparse
import os
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path to search for experiment folders"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        required=True,
        choices=["alpha_blend", "cross_attention", "concat"],
        help="Fusion type to filter folders by"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha value to filter by (only used for alpha_blend fusion type)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run CLIP on"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file to save results (optional)"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load config.json file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def compute_clip_score_single(clip_model, preprocess, image_path, prompt, device):
    """Compute CLIP score for a single image-prompt pair"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Tokenize prompt
        text_input = clip.tokenize([prompt]).to(device)

        # Compute similarity
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            score = torch.nn.functional.cosine_similarity(image_features, text_features).item()

        return score
    except Exception as e:
        print(f"Error computing CLIP score for {image_path}: {e}")
        return None

def main():
    args = parse_args()
    
    # Load CLIP model
    print("Loading CLIP model (ViT-B/32)...")
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    
    # Find all experiment folders
    experiment_folders = []
    for root, dirs, files in os.walk(args.base_path):
        if "config.json" in files:
            experiment_folders.append(root)
    
    print(f"Found {len(experiment_folders)} experiment folders")
    
    # Filter folders based on criteria
    matching_folders = []
    for folder in experiment_folders:
        config_path = os.path.join(folder, "config.json")
        config = load_config(config_path)
        
        if config is None:
            continue
            
        # Check fusion type
        if config.get("fusion_type") != args.fusion_type:
            continue
            
        # Check alpha if specified and fusion type is alpha_blend
        if args.fusion_type == "alpha_blend" and args.alpha is not None:
            if abs(config.get("ref_blend_weight", 0) - args.alpha) > 1e-6:
                continue
        
        matching_folders.append((folder, config))
    
    print(f"Found {len(matching_folders)} matching folders")
    
    if not matching_folders:
        print("No matching folders found!")
        return
    
    # Process each matching folder
    all_scores = []
    results_data = {
        'evaluation_params': {
            'fusion_type': args.fusion_type,
            'alpha': args.alpha,
            'base_path': args.base_path
        },
        'folder_results': [],
        'summary': {}
    }
    
    for folder, config in matching_folders:
        print(f"\nProcessing folder: {folder}")
        
        # Get prompt from config
        prompt = config.get("prompt")
        if not prompt:
            print(f"No prompt found in config for {folder}")
            continue
            
        print(f"Using prompt: '{prompt}'")
        
        # Find all sample images
        samples_folder = os.path.join(folder, "samples")
        if not os.path.exists(samples_folder):
            print(f"Samples folder not found: {samples_folder}")
            continue
            
        sample_files = glob.glob(os.path.join(samples_folder, "*.png")) + \
                      glob.glob(os.path.join(samples_folder, "*.jpg"))
        
        if not sample_files:
            print(f"No sample images found in {samples_folder}")
            continue
            
        print(f"Found {len(sample_files)} sample images")
        
        # Calculate CLIP scores for each sample
        folder_scores = []
        for sample_path in sample_files:
            score = compute_clip_score_single(clip_model, preprocess, sample_path, prompt, args.device)
            if score is not None:
                folder_scores.append(score)
                
                sample_name = os.path.basename(sample_path)
                print(f"  {sample_name}: CLIP = {score:.4f}")
        
        if folder_scores:
            avg_score = sum(folder_scores) / len(folder_scores)
            print(f"  Average CLIP: {avg_score:.4f}")
            all_scores.extend(folder_scores)
            
            # Store results for this folder
            folder_info = {
                'folder': folder,
                'folder_name': os.path.basename(folder),
                'fusion_type': config.get('fusion_type'),
                'alpha': config.get('ref_blend_weight'),
                'prompt': prompt,
                'ref_img': config.get('ref_img'),
                'ref_img_name': os.path.basename(config.get('ref_img', '')),
                'num_samples': len(folder_scores),
                'avg_clip': avg_score,
                'min_clip': min(folder_scores),
                'max_clip': max(folder_scores),
                'scores': folder_scores
            }
            results_data['folder_results'].append(folder_info)
    
    # Print overall statistics and save results
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"\n=== Overall Statistics ===")
        print(f"Total samples processed: {len(all_scores)}")
        print(f"Overall average CLIP: {overall_avg:.4f}")
        print(f"Min CLIP: {min(all_scores):.4f}")
        print(f"Max CLIP: {max(all_scores):.4f}")
        
        # Add summary statistics to results
        results_data['summary'] = {
            'total_samples': len(all_scores),
            'total_folders': len(results_data['folder_results']),
            'overall_avg_clip': overall_avg,
            'overall_min_clip': min(all_scores),
            'overall_max_clip': max(all_scores),
            'overall_std_clip': float(torch.tensor(all_scores).std().item()) if len(all_scores) > 1 else 0.0
        }
    else:
        print("No valid samples processed!")
        results_data['summary'] = {
            'total_samples': 0,
            'total_folders': 0,
            'overall_avg_clip': None,
            'overall_min_clip': None,
            'overall_max_clip': None,
            'overall_std_clip': None
        }
    
    # Save results to JSON file if specified
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()