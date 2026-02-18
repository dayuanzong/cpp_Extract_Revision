"""
Compare 1k3d68 with 2DFAN
Quick test to see current accuracy gap
"""

import sys
import os
from pathlib import Path
import json

# Get root directory
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "sdk"))

from _libs.FaceExtractorWrapper import FaceExtractorWrapper
import cv2
import numpy as np

def calculate_error(landmarks1, landmarks2):
    """Calculate average Euclidean distance between two landmark sets"""
    if len(landmarks1) != len(landmarks2):
        return None
    
    distances = []
    for p1, p2 in zip(landmarks1, landmarks2):
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distances.append(dist)
    
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'median': np.median(distances),
        'distances': distances
    }

def main():
    print("=" * 70)
    print("1k3d68 vs 2DFAN Comparison Test")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Check if 2DFAN model exists
    fan_model_path = models_dir / "FAN" / "2DFAN-4.onnx"
    if not fan_model_path.exists():
        print(f"\n✗ 2DFAN model not found at: {fan_model_path}")
        print("  This test requires 2DFAN-4.onnx in assets/models/FAN/")
        print("\n  Skipping comparison test.")
        return 0
    
    # Find test image
    test_image = root_dir / "aligned_face_0.jpg"
    if not test_image.exists():
        # Try to find any image in tests/test_images
        test_dir = root_dir / "tests" / "test_images"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                test_image = images[0]
    
    if not test_image.exists():
        print("\n✗ No test image found")
        print("  Please add test images to tests/test_images/")
        return 1
    
    print(f"\n✓ Test image: {test_image.name}")
    print(f"✓ 2DFAN model found")
    
    # Load image
    img = cv2.imread(str(test_image))
    if img is None:
        print(f"\n✗ Failed to load image: {test_image}")
        return 1
    
    print(f"✓ Image loaded: {img.shape}")
    
    # Test 1k3d68
    print("\n" + "-" * 70)
    print("Testing 1k3d68...")
    print("-" * 70)
    
    wrapper_1k3d68 = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    faces_1k3d68 = wrapper_1k3d68.extract_faces(img)
    
    if not faces_1k3d68:
        print("✗ 1k3d68 failed to detect faces")
        del wrapper_1k3d68
        return 1
    
    landmarks_1k3d68 = faces_1k3d68[0].get('landmarks', [])
    print(f"✓ Detected {len(landmarks_1k3d68)} landmarks")
    
    del wrapper_1k3d68
    
    # Test 2DFAN
    print("\n" + "-" * 70)
    print("Testing 2DFAN...")
    print("-" * 70)
    
    wrapper_2dfan = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "FAN" / "2DFAN-4.onnx")
    )
    
    faces_2dfan = wrapper_2dfan.extract_faces(img)
    
    if not faces_2dfan:
        print("✗ 2DFAN failed to detect faces")
        del wrapper_2dfan
        return 1
    
    landmarks_2dfan = faces_2dfan[0].get('landmarks', [])
    print(f"✓ Detected {len(landmarks_2dfan)} landmarks")
    
    del wrapper_2dfan
    
    # Compare
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    if len(landmarks_1k3d68) < 68 or len(landmarks_2dfan) < 68:
        print("✗ Insufficient landmarks for comparison")
        return 1
    
    # Compare first 68 points
    error = calculate_error(landmarks_1k3d68[:68], landmarks_2dfan[:68])
    
    if error is None:
        print("✗ Failed to calculate error")
        return 1
    
    print(f"\nError Statistics:")
    print(f"  Mean error:   {error['mean']:.2f} px")
    print(f"  Std error:    {error['std']:.2f} px")
    print(f"  Median error: {error['median']:.2f} px")
    print(f"  Min error:    {error['min']:.2f} px")
    print(f"  Max error:    {error['max']:.2f} px")
    
    # Show per-point errors for worst points
    distances = error['distances']
    worst_indices = np.argsort(distances)[-5:][::-1]
    
    print(f"\nTop 5 worst points:")
    for idx in worst_indices:
        print(f"  Point {idx:2d}: {distances[idx]:.2f} px")
    
    # Visualize if possible
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: 1k3d68 landmarks
        ax = axes[0]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i, (x, y) in enumerate(landmarks_1k3d68[:68]):
            ax.plot(x, y, 'ro', markersize=3)
        ax.set_title('1k3d68 Landmarks')
        ax.axis('off')
        
        # Plot 2: 2DFAN landmarks
        ax = axes[1]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i, (x, y) in enumerate(landmarks_2dfan[:68]):
            ax.plot(x, y, 'go', markersize=3)
        ax.set_title('2DFAN Landmarks (Ground Truth)')
        ax.axis('off')
        
        # Plot 3: Overlay with error visualization
        ax = axes[2]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Normalize distances for color mapping
        max_dist = max(distances)
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(landmarks_1k3d68[:68], landmarks_2dfan[:68])):
            dist = distances[i]
            color_intensity = dist / max_dist
            
            # Draw line from 1k3d68 to 2DFAN
            ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.5, linewidth=1)
            
            # Draw points with color based on error
            ax.plot(x1, y1, 'o', color=(1, 1-color_intensity, 1-color_intensity), markersize=4)
            ax.plot(x2, y2, 'go', markersize=2, alpha=0.5)
        
        ax.set_title(f'Error Visualization (Mean: {error["mean"]:.2f}px)')
        ax.axis('off')
        
        plt.tight_layout()
        
        output_path = root_dir / "tests" / "comparison_result.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved to: {output_path}")
        
    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")
    except Exception as e:
        print(f"\n  (Visualization failed: {e})")
    
    # Save results
    result = {
        'test_image': str(test_image.name),
        'mean_error': error['mean'],
        'std_error': error['std'],
        'median_error': error['median'],
        'min_error': error['min'],
        'max_error': error['max'],
        'num_landmarks': 68
    }
    
    results_file = root_dir / "tests" / "comparison_result.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
