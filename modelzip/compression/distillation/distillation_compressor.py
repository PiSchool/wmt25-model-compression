#!/usr/bin/env python
"""
Knowledge Distillation Compression for WMT25 Model Compression

Implements knowledge distillation methods where a smaller student model
learns from a larger teacher model. Supports various distillation strategies.
"""

import torch
import torch.nn.functional as F
import logging as LOG
from pathlib import Path
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

from ...core.base_compressor import BaseCompressor
from ...core.base_models import BaseModel, HuggingFaceModel
from ...core.experiment_config import ExperimentConfig
from ...constrained_config import COMPRESSION_CONFIG

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DistillationCompressor(BaseCompressor):
    """Knowledge distillation-based compression
    
    Creates a smaller student model that learns from a larger teacher model
    through various distillation techniques (response-based, feature-based).
    """
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Apply knowledge distillation compression
        
        Args:
            model: Teacher model to distill from
            output_path: Directory to save student model
            
        Returns:
            BaseModel: Compressed student model wrapper
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get distillation configuration
        distill_config = self.config.compression_params.get(
            "distillation",
            COMPRESSION_CONFIG["distillation"]["response_based"]
        )
        
        # Create student model
        student_model = self._create_student_model(model, distill_config)
        
        # Train student model with distillation
        if distill_config.get("train_student", True):
            trained_student = self._train_student_model(
                teacher=model,
                student=student_model,
                config=distill_config
            )
        else:
            trained_student = student_model
        
        # Save student model and tokenizer
        trained_student.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)
        
        # Copy submission script and save metadata
        self._copy_run_script(output_path)
        self._save_compression_metadata(output_path, {
            "distillation_config": distill_config,
            "student_architecture": distill_config.get("student_architecture", "reduced"),
            "teacher_size_mb": model.get_model_size(),
            "distillation_method": distill_config.get("method", "response_based")
        })
        
        # Create new model wrapper
        compressed_config = ExperimentConfig(
            name=f"{self.config.name}_distilled",
            compression_method=self.config.compression_method,
            lang_pair=self.config.lang_pair,
            compression_params={"distillation": distill_config}
        )
        
        return HuggingFaceModel(output_path, compressed_config)
    
    def _create_student_model(self, teacher_model: BaseModel, distill_config: dict):
        """Create student model architecture
        
        Args:
            teacher_model: Teacher model to learn from
            distill_config: Distillation configuration
            
        Returns:
            Student model with reduced architecture
        """
        architecture = distill_config.get("student_architecture", "reduced")
        
        if architecture == "reduced":
            return self._create_reduced_student(teacher_model, distill_config)
        elif architecture == "custom":
            return self._create_custom_student(teacher_model, distill_config)
        else:
            LOG.warning(f"Unknown student architecture: {architecture}, using reduced")
            return self._create_reduced_student(teacher_model, distill_config)
    
    def _create_reduced_student(self, teacher_model: BaseModel, distill_config: dict):
        """Create student with reduced layers/dimensions
        
        Args:
            teacher_model: Teacher model
            distill_config: Configuration parameters
            
        Returns:
            Student model with reduced architecture
        """
        # Get teacher configuration
        teacher_config = teacher_model.model.config
        
        # Create reduced configuration
        student_config = type(teacher_config)(**teacher_config.to_dict())
        
        # Reduce model dimensions
        reduction_factor = distill_config.get("reduction_factor", 0.5)
        
        if hasattr(student_config, 'num_hidden_layers'):
            student_config.num_hidden_layers = max(
                1, int(student_config.num_hidden_layers * reduction_factor)
            )
        
        if hasattr(student_config, 'hidden_size'):
            student_config.hidden_size = max(
                256, int(student_config.hidden_size * reduction_factor)
            )
        
        if hasattr(student_config, 'intermediate_size'):
            student_config.intermediate_size = max(
                512, int(student_config.intermediate_size * reduction_factor)
            )
        
        # Create student model with reduced configuration
        student_model = AutoModelForCausalLM.from_config(student_config)
        
        LOG.info(f"Created student model with {student_config.num_hidden_layers} layers "
                f"and {student_config.hidden_size} hidden size")
        
        return student_model
    
    def _create_custom_student(self, teacher_model: BaseModel, distill_config: dict):
        """Create student with custom architecture
        
        Args:
            teacher_model: Teacher model
            distill_config: Configuration with custom parameters
            
        Returns:
            Student model with custom architecture
        """
        # For now, fallback to reduced student
        # In practice, this could load a specific smaller model
        LOG.info("Custom student architecture not implemented, using reduced")
        return self._create_reduced_student(teacher_model, distill_config)
    
    def _train_student_model(self, teacher: BaseModel, student, config: dict):
        """Train student model using knowledge distillation
        
        Args:
            teacher: Teacher model
            student: Student model to train
            config: Training configuration
            
        Returns:
            Trained student model
        """
        # Create synthetic training data or use actual data
        training_data = self._prepare_distillation_data(config)
        
        if not training_data:
            LOG.warning("No training data available for distillation")
            return student
        
        # Create custom trainer for distillation
        trainer = self._create_distillation_trainer(
            teacher=teacher,
            student=student,
            training_data=training_data,
            config=config
        )
        
        # Train the student model
        LOG.info("Starting knowledge distillation training...")
        trainer.train()
        
        return student
    
    def _prepare_distillation_data(self, config: dict):
        """Prepare data for distillation training
        
        Args:
            config: Distillation configuration
            
        Returns:
            Training dataset for distillation
        """
        # For now, return None to skip training
        # In practice, this would prepare actual training data
        return None
    
    def _create_distillation_trainer(self, teacher, student, training_data, config):
        """Create trainer for knowledge distillation
        
        Args:
            teacher: Teacher model
            student: Student model
            training_data: Training dataset
            config: Training configuration
            
        Returns:
            Trainer configured for distillation
        """
        training_args = TrainingArguments(
            output_dir="./distillation_training",
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            learning_rate=config.get("learning_rate", 5e-5),
            logging_steps=10,
            save_steps=500,
        )
        
        # Custom trainer would implement distillation loss
        trainer = Trainer(
            model=student,
            args=training_args,
            train_dataset=training_data,
        )
        
        return trainer
    
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Calculate compression ratio for distilled model
        
        Args:
            original_model: Original teacher model
            compressed_model: Compressed student model
            
        Returns:
            float: Compression ratio (teacher_size / student_size)
        """
        teacher_size = original_model.get_model_size()
        student_size = compressed_model.get_model_size()
        return teacher_size / student_size if student_size > 0 else 0.0 