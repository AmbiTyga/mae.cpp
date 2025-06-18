#!/usr/bin/env python3
"""
MAE Server Client - Demonstrates all endpoints including masking and reconstruction
"""

import requests
import base64
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

class MAEClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        
    def encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def decode_image(self, base64_str):
        """Decode base64 to image"""
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def health_check(self):
        """Check server health"""
        response = requests.get(f"{self.server_url}/health")
        return response.json()
    
    def get_info(self):
        """Get server info"""
        response = requests.get(f"{self.server_url}/info")
        return response.json()
    
    def visualize_patches(self, image_path, show_numbers=True):
        """Visualize how image is divided into patches"""
        payload = {
            "image": self.encode_image(image_path),
            "show_numbers": show_numbers
        }
        
        response = requests.post(f"{self.server_url}/visualize_patches", json=payload)
        result = response.json()
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        # Decode result image
        patched_img = self.decode_image(result["patched_image"])
        
        print(f"Patch size: {result['patch_size']}x{result['patch_size']}")
        print(f"Patches per side: {result['num_patches_per_side']}")
        print(f"Total patches: {result['total_patches']}")
        
        return patched_img, result
    
    def mask_image(self, image_path, mask_ratio=0.75):
        """Create masked version of image"""
        payload = {
            "image": self.encode_image(image_path),
            "mask_ratio": mask_ratio
        }
        
        response = requests.post(f"{self.server_url}/mask_image", json=payload)
        result = response.json()
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None, None
        
        # Decode masked image
        masked_img = self.decode_image(result["masked_image"])
        
        print(f"Mask ratio: {result['mask_ratio']}")
        print(f"Masked patches: {result['num_masked_patches']}")
        print(f"Visible patches: {result['num_visible_patches']}")
        
        return masked_img, result
    
    def reconstruct(self, image_path, mask_ratio=0.75):
        """Reconstruct image with given mask ratio"""
        payload = {
            "image": self.encode_image(image_path),
            "mask_ratio": mask_ratio
        }
        
        response = requests.post(f"{self.server_url}/reconstruct", json=payload)
        result = response.json()
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        # Decode reconstructed image
        reconstructed_img = self.decode_image(result["reconstruction"])
        
        print(f"Processing time: {result['processing_time_ms']}ms")
        
        return reconstructed_img
    
    def demo_workflow(self, image_path, mask_ratios=[0.5, 0.75, 0.9]):
        """Complete demo workflow showing all capabilities"""
        print(f"\n=== MAE Demo for {image_path} ===\n")
        
        # Load original image
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Create figure for results
        num_ratios = len(mask_ratios)
        fig, axes = plt.subplots(3, num_ratios + 1, figsize=(4*(num_ratios+1), 12))
        
        # Show original
        for i in range(3):
            axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Original')
        
        # Process each mask ratio
        for idx, ratio in enumerate(mask_ratios):
            print(f"\nProcessing with mask_ratio={ratio}")
            
            # Get masked image
            masked_img, mask_info = self.mask_image(image_path, ratio)
            if masked_img is not None:
                axes[0, idx+1].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
                axes[0, idx+1].set_title(f'Masked ({int(ratio*100)}%)')
                axes[0, idx+1].axis('off')
                
                # Show mask pattern
                mask_array = np.array(mask_info['mask'])
                axes[1, idx+1].imshow(mask_array, cmap='gray', vmin=0, vmax=1)
                axes[1, idx+1].set_title(f'Mask Pattern')
                axes[1, idx+1].axis('off')
            
            # Get reconstruction
            reconstructed = self.reconstruct(image_path, ratio)
            if reconstructed is not None:
                axes[2, idx+1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
                axes[2, idx+1].set_title(f'Reconstructed')
                axes[2, idx+1].axis('off')
        
        # Add row labels
        axes[0, 0].text(-0.1, 0.5, 'Input/Masked', transform=axes[0, 0].transAxes, 
                        fontsize=12, rotation=90, va='center', ha='right')
        axes[1, 0].text(-0.1, 0.5, 'Mask Pattern', transform=axes[1, 0].transAxes, 
                        fontsize=12, rotation=90, va='center', ha='right')
        axes[2, 0].text(-0.1, 0.5, 'Reconstruction', transform=axes[2, 0].transAxes, 
                        fontsize=12, rotation=90, va='center', ha='right')
        
        plt.suptitle(f'MAE Reconstruction Demo: {Path(image_path).name}', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = f"mae_demo_{Path(image_path).stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved demo to {output_path}")
        plt.show()
    
    def save_individual_results(self, image_path, mask_ratio=0.75, output_dir="mae_results"):
        """Save individual masked and reconstructed images"""
        Path(output_dir).mkdir(exist_ok=True)
        
        stem = Path(image_path).stem
        
        # Get patch visualization
        print("\nVisualizing patches...")
        patched_img, _ = self.visualize_patches(image_path)
        if patched_img is not None:
            cv2.imwrite(f"{output_dir}/{stem}_patches.png", patched_img)
            print(f"Saved: {output_dir}/{stem}_patches.png")
        
        # Get masked image
        print("\nCreating masked image...")
        masked_img, mask_info = self.mask_image(image_path, mask_ratio)
        if masked_img is not None:
            cv2.imwrite(f"{output_dir}/{stem}_masked_{int(mask_ratio*100)}.png", masked_img)
            print(f"Saved: {output_dir}/{stem}_masked_{int(mask_ratio*100)}.png")
            
            # Save mask as grayscale image
            mask_array = np.array(mask_info['mask']) * 255
            cv2.imwrite(f"{output_dir}/{stem}_mask_{int(mask_ratio*100)}.png", mask_array.astype(np.uint8))
            print(f"Saved: {output_dir}/{stem}_mask_{int(mask_ratio*100)}.png")
        
        # Get reconstruction
        print("\nReconstructing image...")
        reconstructed = self.reconstruct(image_path, mask_ratio)
        if reconstructed is not None:
            cv2.imwrite(f"{output_dir}/{stem}_reconstructed_{int(mask_ratio*100)}.png", reconstructed)
            print(f"Saved: {output_dir}/{stem}_reconstructed_{int(mask_ratio*100)}.png")
        
        print(f"\nAll results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='MAE Server Client')
    parser.add_argument('--server', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--action', choices=['patches', 'mask', 'reconstruct', 'demo', 'save_all'], 
                        default='demo', help='Action to perform')
    parser.add_argument('--mask-ratio', type=float, default=0.75, help='Mask ratio (0-1)')
    parser.add_argument('--output-dir', default='mae_results', help='Output directory for saved results')
    
    args = parser.parse_args()
    
    # Create client
    client = MAEClient(args.server)
    
    # Check server health
    print("Checking server health...")
    health = client.health_check()
    print(f"Server status: {health}")
    
    # Get server info
    info = client.get_info()
    print(f"Server info: {info}")
    
    # Perform requested action
    if args.action == 'patches':
        img, _ = client.visualize_patches(args.image)
        if img is not None:
            cv2.imshow('Patches Visualization', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.action == 'mask':
        img, _ = client.mask_image(args.image, args.mask_ratio)
        if img is not None:
            cv2.imshow('Masked Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.action == 'reconstruct':
        img = client.reconstruct(args.image, args.mask_ratio)
        if img is not None:
            cv2.imshow('Reconstructed Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.action == 'demo':
        client.demo_workflow(args.image, mask_ratios=[0.5, 0.75, 0.9])
    
    elif args.action == 'save_all':
        client.save_individual_results(args.image, args.mask_ratio, args.output_dir)

if __name__ == "__main__":
    main()