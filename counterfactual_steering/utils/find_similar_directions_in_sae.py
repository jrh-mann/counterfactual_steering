#!/usr/bin/env python
"""
Find SAE features most similar to steering vectors at each layer.

Usage:
    python find_similar_features.py --steering_vectors_path steering_vectors.pt
    
Note: This script downloads SAE weights from HuggingFace (andyrdt/saes-gpt-oss-20b).
The weights are separate from the MAE cache which only contains pre-computed examples.
"""
import argparse
import os
import sys
import torch
from typing import List, Tuple, Dict
from itertools import combinations
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary
import matplotlib.pyplot as plt
import numpy as np


def load_steering_vectors(path: str) -> torch.Tensor:
    """Load steering vectors from .pt file."""
    steering = torch.load(path, map_location="cpu")
    if isinstance(steering, dict):
        # If it's a dict, try to extract the tensor
        if "steering_vectors" in steering:
            steering = steering["steering_vectors"]
        else:
            # Take the first tensor value
            steering = next(iter(steering.values()))
    return steering.to(torch.float32).to("cpu")


def load_sae_decoder(layer: int, trainer: int, sae_repo: str = "andyrdt/saes-gpt-oss-20b") -> torch.Tensor:
    """
    Load SAE decoder weights for a specific layer and trainer.
    
    Downloads from HuggingFace and uses dictionary_learning to load the SAE.
    Returns the decoder weight matrix with shape (d_model, n_features).
    """
    print(f"  Loading SAE weights from HuggingFace ({sae_repo})...")
    subfolder = f"resid_post_layer_{layer}/trainer_{trainer}"
    
    # Download SAE weights (uses HF cache, won't re-download if already present)
    local_dir = snapshot_download(
        repo_id=sae_repo,
        allow_patterns=f"{subfolder}/*",
        local_dir_use_symlinks=False,
        cache_dir=None  # Uses default ~/.cache/huggingface/hub/
    )
    
    final_path = os.path.join(local_dir, subfolder)
    sae, config = load_dictionary(final_path, device="cpu")
    
    # Extract decoder weights
    # The BatchTopKSAE has a decoder which is a nn.Linear module
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
        W_dec = sae.decoder.weight.detach().cpu()
        # decoder.weight has shape (d_model, n_features)
        # Each column represents a feature's direction in model space
    elif hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec.detach().cpu()
    else:
        raise AttributeError(f"Could not find decoder weights in SAE. Available attributes: {dir(sae)}")
    
    print(f"  SAE decoder shape: {W_dec.shape} (d_model x n_features)")
    return W_dec


def compute_cosine_similarity(vec: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between a vector and all columns of a matrix.
    
    Args:
        vec: shape (d_model,)
        matrix: shape (d_model, n_features)
    
    Returns:
        similarities: shape (n_features,)
    """
    # Normalize vector
    vec_norm = vec / (vec.norm() + 1e-8)
    
    # Normalize each feature (column)
    matrix_norm = matrix / (matrix.norm(dim=0, keepdim=True) + 1e-8)
    
    # Compute dot product
    # ensure same dtype and device (float32 and cuda:0) 
    similarities = vec_norm @ matrix_norm
    
    return similarities


def find_top_k_features(
    steering_vector: torch.Tensor,
    decoder: torch.Tensor,
    k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar features to the steering vector.
    
    Returns:
        List of (feature_id, similarity_score) tuples
    """
    similarities = compute_cosine_similarity(steering_vector, decoder)
    top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
    
    results = [
        (int(idx), float(sim)) 
        for idx, sim in zip(top_k_indices.indices, top_k_indices.values)
    ]
    
    return results


def try_reconstruct(
    steering_vector: torch.Tensor,
    layer: int,
    trainer: int = 0,
    sae_repo: str = "andyrdt/saes-gpt-oss-20b",
    top_k: int = 20,
    max_combination_size: int = 5
) -> Dict:
    """
    Try to reconstruct the steering vector using combinations of top-k SAE features.
    
    For each combination size from 1 to max_combination_size, try all combinations
    of features from the top-k most similar features. Return the combination that
    maximizes cosine similarity with the original steering vector.
    
    Args:
        steering_vector: The target vector to reconstruct (d_model,)
        decoder: SAE decoder weight matrix (d_model, n_features)
        top_k: Number of top similar features to consider
        max_combination_size: Maximum number of features to combine (n)
    
    Returns:
        Dictionary containing:
            - best_combination: List of feature indices in the best combination
            - best_similarity: Cosine similarity achieved
            - best_reconstruction: The reconstructed vector
            - all_results: List of all tried combinations with their scores
    """
    # First, find top-k most similar features
    print(f"  Loading decoder weights...")
    decoder = load_sae_decoder(layer, trainer, sae_repo)

    print(f"  Finding top {top_k} similar features...")
    top_features = find_top_k_features(steering_vector, decoder, k=top_k)
    feature_indices = [feat_id for feat_id, _ in top_features]
    
    print(f"  Trying all combinations up to size {max_combination_size}...")
    
    best_similarity = -1.0
    best_combination = []
    best_reconstruction = None
    all_results = []
    
    total_combinations = sum(len(list(combinations(feature_indices, r))) 
                            for r in range(1, max_combination_size + 1))
    print(f"  Total combinations to try: {total_combinations}")
    
    # Try all combination sizes from 1 to max_combination_size
    for combo_size in range(1, max_combination_size + 1):
        for combo in combinations(feature_indices, combo_size):
            # Sum the feature directions
            combo_list = list(combo)
            reconstruction = decoder[:, combo_list].sum(dim=1)
            
            # Compute cosine similarity with original steering vector
            similarity = float(compute_cosine_similarity(steering_vector, reconstruction.unsqueeze(1)).squeeze())
            
            result = {
                'combination': combo_list,
                'size': combo_size,
                'similarity': similarity
            }
            all_results.append(result)
            
            # Track best
            if similarity > best_similarity:
                best_similarity = similarity
                best_combination = combo_list
                best_reconstruction = reconstruction
    
    # Sort all results by similarity
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'best_combination': best_combination,
        'best_similarity': best_similarity,
        'best_reconstruction': best_reconstruction,
        'all_results': all_results,
        'top_10_results': all_results[:10]
    }


def reconstruction_analysis(
    steering_vector: torch.Tensor,
    layer: int,
    trainer: int = 0,
    sae_repo: str = "andyrdt/saes-gpt-oss-20b",
    top_k: int = 20,
    max_n: int = 10,
    plot: bool = True,
    save_path: str = None
) -> Dict:
    """
    Analyze how reconstruction quality improves with combination size.
    
    Tests all combination sizes from 1 to max_n and plots the best similarity
    achieved at each size.
    
    Args:
        steering_vector: The target vector to reconstruct (d_model,)
        layer: Layer number for SAE
        trainer: Trainer index (0 or 1)
        sae_repo: HuggingFace repo for SAE weights
        top_k: Number of top features to consider
        max_n: Maximum combination size to test
        plot: Whether to display the plot
        save_path: Optional path to save the plot
    
    Returns:
        Dictionary containing:
            - sizes: List of combination sizes tested [1, 2, 3, ...]
            - best_similarities: Best similarity achieved at each size
            - best_combinations: Best feature combination at each size
            - all_results: All reconstruction results for each size
    """
    print(f"  Loading decoder weights for layer {layer}...")
    decoder = load_sae_decoder(layer, trainer, sae_repo)
    
    print(f"  Finding top {top_k} similar features...")
    top_features = find_top_k_features(steering_vector, decoder, k=top_k)
    feature_indices = [feat_id for feat_id, _ in top_features]
    
    sizes = []
    best_similarities = []
    best_combinations = []
    all_results_by_size = []
    
    print(f"\n  Testing combination sizes 1 to {max_n}...")
    for n in range(1, max_n + 1):
        print(f"    Size {n}:", end=" ")
        
        best_sim = -1.0
        best_combo = []
        size_results = []
        
        num_combos = len(list(combinations(feature_indices, n)))
        print(f"trying {num_combos} combinations...", end=" ")
        
        for combo in combinations(feature_indices, n):
            combo_list = list(combo)
            reconstruction = decoder[:, combo_list].sum(dim=1)
            similarity = float(compute_cosine_similarity(steering_vector, reconstruction.unsqueeze(1)).squeeze())
            
            result = {
                'combination': combo_list,
                'size': n,
                'similarity': similarity
            }
            size_results.append(result)
            
            if similarity > best_sim:
                best_sim = similarity
                best_combo = combo_list
        
        sizes.append(n)
        best_similarities.append(best_sim)
        best_combinations.append(best_combo)
        all_results_by_size.append(size_results)
        
        print(f"best: {best_sim:.6f} {best_combo}")
    
    # Create plot
    if plot or save_path:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, best_similarities, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Combination Size (n)', fontsize=12)
        plt.ylabel('Best Cosine Similarity', fontsize=12)
        plt.title(f'Reconstruction Quality vs Combination Size\nLayer {layer}, Top-{top_k} features', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(sizes)
        
        # Add value labels on points
        for i, (s, sim) in enumerate(zip(sizes, best_similarities)):
            plt.annotate(f'{sim:.4f}', 
                        xy=(s, sim), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved to: {save_path}")
        
        if plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'sizes': sizes,
        'best_similarities': best_similarities,
        'best_combinations': best_combinations,
        'all_results_by_size': all_results_by_size,
        'layer': layer,
        'top_k': top_k
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find SAE features most similar to steering vectors"
    )
    parser.add_argument(
        "--steering_vectors_path",
        type=str,
        required=True,
        help="Path to steering_vectors.pt file (shape: [24, 2880])"
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default="andyrdt/saes-gpt-oss-20b",
        help="HuggingFace repo containing SAE weights"
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=0,
        help="Trainer index (0=k64 or 1=k128)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top similar features to report per layer"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="3,7,11,15,19,23",
        help="Comma-separated list of layers to analyze"
    )
    parser.add_argument(
        "--mae_cache_dir",
        type=str,
        default=".mae_cache",
        help="Directory where MAE data is cached (for dashboard URLs)"
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Run reconstruction mode: find best combination of features to reconstruct steering vector"
    )
    parser.add_argument(
        "--reconstruct_top_k",
        type=int,
        default=20,
        help="Number of top features to consider for reconstruction (default: 20)"
    )
    parser.add_argument(
        "--max_combination_size",
        type=int,
        default=5,
        help="Maximum number of features to combine in reconstruction (default: 5)"
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run reconstruction analysis: test all sizes from 1 to max_combination_size and plot results"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save analysis plot (e.g., 'reconstruction_analysis.png')"
    )
    
    args = parser.parse_args()
    
    # Load steering vectors
    print(f"Loading steering vectors from {args.steering_vectors_path}...")
    steering_vectors = load_steering_vectors(args.steering_vectors_path)
    print(f"Steering vectors shape: {steering_vectors.shape}")
    
    if steering_vectors.shape[0] != 24:
        print(f"Warning: Expected 24 layers, got {steering_vectors.shape[0]}")
    
    layers = [int(x.strip()) for x in args.layers.split(",")]
    
    print(f"\nAnalyzing layers: {layers}")
    print(f"SAE repo: {args.sae_repo}")
    print(f"Trainer: {args.trainer}")
    print(f"Top-K: {args.top_k}\n")
    print("=" * 80)
    
    # Process each layer
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}")
        print("-" * 80)
        
        try:
            # Get steering vector for this layer
            if layer_idx >= steering_vectors.shape[0]:
                print(f"Warning: Layer {layer_idx} not in steering vectors (max: {steering_vectors.shape[0]-1})")
                continue
            
            steering_vec = steering_vectors[layer_idx]
            print(f"Steering vector shape: {steering_vec.shape}")
            
            if args.analysis:
                # Run reconstruction analysis mode (test all sizes and plot)
                print(f"  Running reconstruction analysis...")
                save_path = None
                if args.save_plot:
                    save_path = args.save_plot.replace('.png', f'_layer{layer_idx}.png')
                
                analysis_results = reconstruction_analysis(
                    steering_vec,
                    layer=layer_idx,
                    trainer=args.trainer,
                    sae_repo=args.sae_repo,
                    top_k=args.reconstruct_top_k,
                    max_n=args.max_combination_size,
                    plot=True,
                    save_path=save_path
                )
                
                print(f"\n  ANALYSIS SUMMARY:")
                print(f"    Tested sizes: 1 to {args.max_combination_size}")
                print(f"    Best similarity: {max(analysis_results['best_similarities']):.6f} (at size {analysis_results['sizes'][analysis_results['best_similarities'].index(max(analysis_results['best_similarities']))]})")
                
                print(f"\n  Best combination at each size:")
                for size, sim, combo in zip(analysis_results['sizes'], 
                                            analysis_results['best_similarities'],
                                            analysis_results['best_combinations']):
                    feat_str = ','.join(str(f) for f in combo)
                    print(f"    Size {size}: sim={sim:.6f} features=[{feat_str}]")
                
            elif args.reconstruct:
                # Run reconstruction mode
                print(f"  Running reconstruction...")
                results = try_reconstruct(
                    steering_vec,
                    layer=layer_idx,
                    trainer=args.trainer,
                    sae_repo=args.sae_repo,
                    top_k=args.reconstruct_top_k,
                    max_combination_size=args.max_combination_size
                )
                
                print(f"\n  BEST RECONSTRUCTION:")
                print(f"    Features: {results['best_combination']}")
                print(f"    Combination size: {len(results['best_combination'])}")
                print(f"    Cosine similarity: {results['best_similarity']:.6f}")
                
                print(f"\n  Top 10 combinations:")
                for rank, result in enumerate(results['top_10_results'], 1):
                    feat_str = ','.join(str(f) for f in result['combination'])
                    print(f"    {rank:2d}. [{feat_str}] (size={result['size']}, sim={result['similarity']:.6f})")
                
                # Provide URL to view best combination
                feat_ids = ",".join(str(f) for f in results['best_combination'])
                print(f"\n  View best combination in dashboard:")
                print(f"    http://localhost:7863/?model=gpt&layer={layer_idx}&trainer={args.trainer}&fids={feat_ids}")
                
            else:
                # Original mode: just find top-k similar features
                # Load SAE decoder for this layer
                decoder = load_sae_decoder(layer_idx, args.trainer, args.sae_repo)
                
                # Verify dimensions match
                if decoder.shape[0] != steering_vec.shape[0]:
                    print(f"  Error: Dimension mismatch!")
                    print(f"    Decoder shape: {decoder.shape} (expected: [{steering_vec.shape[0]}, n_features])")
                    print(f"    Steering vector shape: {steering_vec.shape}")
                    continue
                
                print(f"  Computing cosine similarities with {decoder.shape[1]} features...")
                top_features = find_top_k_features(steering_vec, decoder, k=args.top_k)
                
                print(f"\n  Top {args.top_k} most similar features:")
                for rank, (feat_id, similarity) in enumerate(top_features, 1):
                    print(f"    {rank:2d}. Feature {feat_id:5d}: similarity = {similarity:7.4f}")
                
                # Provide URL to view features
                feat_ids = ",".join(str(f[0]) for f in top_features[:5])
                print(f"\n  View top 5 in dashboard:")
                print(f"    http://localhost:7863/?model=gpt&layer={layer_idx}&trainer={args.trainer}&fids={feat_ids}")
            
        except Exception as e:
            print(f"  Error processing layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())