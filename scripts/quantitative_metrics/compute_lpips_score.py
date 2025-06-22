import torch
import lpips
from PIL import Image
import argparse
import os
import json
import glob
from torchvision import transforms

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
        "--backbone",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run LPIPS on"
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

def preprocess_image(image_path, device, target_size=(768, 768)):
    """Load and preprocess image for LPIPS"""
    try:
        image = Image.open(image_path).convert("RGB")
        # LPIPS expects images in [-1, 1] range and same dimensions
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Convert [0,1] to [-1,1]
        ])
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def compute_lpips_score(lpips_model, image1_tensor, image2_tensor):
    """Compute LPIPS score between two image tensors"""
    with torch.no_grad():
        score = lpips_model(image1_tensor, image2_tensor).item()
    return score

def main():
    args = parse_args()
    
    # Initialize LPIPS model
    print(f"Loading LPIPS model with {args.backbone} backbone...")
    lpips_model = lpips.LPIPS(net=args.backbone).to(args.device)
    
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
            'backbone': args.backbone,
            'base_path': args.base_path
        },
        'folder_results': [],
        'summary': {}
    }
    
    for folder, config in matching_folders:
        print(f"\nProcessing folder: {folder}")
        
        # Get reference image path
        ref_img_path = config.get("ref_img")
        if not ref_img_path:
            print(f"No ref_img found in config for {folder}")
            continue
        
        if not os.path.exists(ref_img_path):
            print(f"Reference image not found: {ref_img_path}")
            continue
            
        # Load reference image
        ref_image_tensor = preprocess_image(ref_img_path, args.device)
        if ref_image_tensor is None:
            continue
            
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
        
        # Calculate LPIPS for each sample
        folder_scores = []
        for sample_path in sample_files:
            sample_tensor = preprocess_image(sample_path, args.device)
            if sample_tensor is None:
                continue
                
            score = compute_lpips_score(lpips_model, ref_image_tensor, sample_tensor)
            folder_scores.append(score)
            
            sample_name = os.path.basename(sample_path)
            print(f"  {sample_name}: LPIPS = {score:.4f}")
        
        if folder_scores:
            avg_score = sum(folder_scores) / len(folder_scores)
            print(f"  Average LPIPS: {avg_score:.4f}")
            all_scores.extend(folder_scores)
            
            # Store results for this folder
            folder_info = {
                'folder': folder,
                'folder_name': os.path.basename(folder),
                'fusion_type': config.get('fusion_type'),
                'alpha': config.get('ref_blend_weight'),
                'ref_img': config.get('ref_img'),
                'ref_img_name': os.path.basename(config.get('ref_img', '')),
                'num_samples': len(folder_scores),
                'avg_lpips': avg_score,
                'min_lpips': min(folder_scores),
                'max_lpips': max(folder_scores),
                'scores': folder_scores
            }
            results_data['folder_results'].append(folder_info)
    
    # Print overall statistics and save results
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"\n=== Overall Statistics ===")
        print(f"Total samples processed: {len(all_scores)}")
        print(f"Overall average LPIPS: {overall_avg:.4f}")
        print(f"Min LPIPS: {min(all_scores):.4f}")
        print(f"Max LPIPS: {max(all_scores):.4f}")
        
        # Add summary statistics to results
        results_data['summary'] = {
            'total_samples': len(all_scores),
            'total_folders': len(results_data['folder_results']),
            'overall_avg_lpips': overall_avg,
            'overall_min_lpips': min(all_scores),
            'overall_max_lpips': max(all_scores),
            'overall_std_lpips': float(torch.tensor(all_scores).std().item()) if len(all_scores) > 1 else 0.0
        }
    else:
        print("No valid samples processed!")
        results_data['summary'] = {
            'total_samples': 0,
            'total_folders': 0,
            'overall_avg_lpips': None,
            'overall_min_lpips': None,
            'overall_max_lpips': None,
            'overall_std_lpips': None
        }
    
    # Save results to JSON file if specified
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main() 