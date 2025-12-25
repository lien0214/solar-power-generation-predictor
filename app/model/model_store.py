"""Model storage and loading utilities."""

import os
import pickle
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelStore:
    """Manages saving and loading of trained models."""
    
    def __init__(self, model_dir: str = "./models"):
        """Initialize model store.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelStore initialized: {self.model_dir}")
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name: Name for the model (e.g., 'weather', 'solar')
            metadata: Optional metadata dict (features, params, etc.)
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.model_dir / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save metadata if provided
        if metadata:
            meta_path = self.model_dir / f"{model_name}_metadata.pkl"
            metadata['saved_at'] = timestamp
            joblib.dump(metadata, meta_path)
            logger.info(f"Metadata saved: {meta_path}")
        
        return str(model_path)
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object or None if not found
        """
        model_path = self.model_dir / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def load_metadata(self, model_name: str) -> Optional[Dict]:
        """Load model metadata.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Metadata dict or None if not found
        """
        meta_path = self.model_dir / f"{model_name}_metadata.pkl"
        
        if not meta_path.exists():
            return None
        
        try:
            metadata = joblib.load(meta_path)
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata for {model_name}: {e}")
            return None
    
    def list_available_models(self) -> List[str]:
        """List all available models.
        
        Returns:
            List of model names
        """
        model_files = list(self.model_dir.glob("*_model.pkl"))
        model_names = [f.stem.replace('_model', '') for f in model_files]
        return sorted(model_names)
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model exists, False otherwise
        """
        model_path = self.model_dir / f"{model_name}_model.pkl"
        return model_path.exists()
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model and its metadata.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self.model_dir / f"{model_name}_model.pkl"
            meta_path = self.model_dir / f"{model_name}_metadata.pkl"
            
            deleted = False
            if model_path.exists():
                model_path.unlink()
                deleted = True
                logger.info(f"Deleted model: {model_path}")
            
            if meta_path.exists():
                meta_path.unlink()
                logger.info(f"Deleted metadata: {meta_path}")
            
            return deleted
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'name': model_name,
            'exists': self.model_exists(model_name),
            'metadata': None,
            'file_size': None,
            'modified_time': None
        }
        
        model_path = self.model_dir / f"{model_name}_model.pkl"
        if model_path.exists():
            info['file_size'] = model_path.stat().st_size
            info['modified_time'] = datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat()
        
        metadata = self.load_metadata(model_name)
        if metadata:
            info['metadata'] = metadata
        
        return info
