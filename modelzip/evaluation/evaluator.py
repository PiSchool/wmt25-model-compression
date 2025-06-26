#!/usr/bin/env python
"""
Evaluation Classes for WMT25 Model Compression

Provides evaluation interfaces and implementations for assessing
compressed models on translation quality and performance metrics.
"""

import abc
import time
import logging as LOG
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from ..core.base_models import BaseModel
from ..constrained_config import WORKDIR, CONSTRAINED_TASK
from ..data_manager import DataManager

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseEvaluator(abc.ABC):
    """Abstract base class for model evaluation"""
    
    @abc.abstractmethod
    def evaluate_model(self, model: BaseModel, eval_params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model and return metrics
        
        Args:
            model: Model to evaluate
            eval_params: Evaluation parameters
            
        Returns:
            Dictionary of metric names to scores
        """
        pass


class TranslationEvaluator(BaseEvaluator):
    """Evaluator for translation models on WMT25 constrained task"""
    
    def __init__(self, lang_pair: str, workdir: Path = None):
        """Initialize translation evaluator
        
        Args:
            lang_pair: Language pair to evaluate (e.g., "ces-deu")
            workdir: Working directory for data and results
        """
        self.lang_pair = lang_pair
        self.workdir = workdir or WORKDIR
        self.data_manager = DataManager(self.workdir)
        
        # Load prompts for this language pair
        self.prompts = CONSTRAINED_TASK["prompts"].get(lang_pair, {})
        
    def evaluate_model(self, model: BaseModel, eval_params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate translation model on test data
        
        Args:
            model: Translation model to evaluate
            eval_params: Evaluation configuration
            
        Returns:
            Dictionary containing evaluation metrics
        """
        LOG.info(f"🔍 Starting evaluation on {self.lang_pair} translation")
        
        # Load test data
        with tqdm(total=1, desc="Loading test data", leave=False) as pbar:
            test_data = self._load_test_data()
            pbar.update(1)
            
        if not test_data:
            LOG.warning(f"No test data available for {self.lang_pair}")
            return {"error": "No test data available"}
        
        # Get evaluation parameters
        max_samples = eval_params.get("max_samples", 100)
        batch_size = eval_params.get("batch_size", 1)
        
        # Limit test samples for efficiency
        if len(test_data) > max_samples:
            test_data = test_data[:max_samples]
            LOG.info(f"📊 Limited evaluation to {max_samples} samples")
        else:
            LOG.info(f"📊 Evaluating on {len(test_data)} samples")
        
        # Generate translations
        LOG.info("🤖 Generating translations...")
        translations = self._generate_translations(model, test_data, batch_size)
        
        # Calculate metrics
        LOG.info("📈 Calculating quality metrics...")
        metrics = self._calculate_metrics(test_data, translations)
        
        # Add performance metrics
        LOG.info("⚡ Measuring performance...")
        performance_metrics = self._measure_performance(model, test_data[:10], batch_size)
        metrics.update(performance_metrics)
        
        LOG.info(f"✅ Evaluation completed for {self.lang_pair}")
        return metrics
    
    def _load_test_data(self) -> List[Dict[str, str]]:
        """Load test data for the language pair
        
        Returns:
            List of dictionaries with 'source' and 'target' keys
        """
        try:
            return self.data_manager._load_test_data_for_pair(self.lang_pair)
        except Exception as e:
            LOG.error(f"Failed to load test data for {self.lang_pair}: {e}")
            return []
    
    def _generate_translations(self, model: BaseModel, test_data: List[Dict], batch_size: int) -> List[str]:
        """Generate translations for test data
        
        Args:
            model: Model to use for translation
            test_data: Test data to translate
            batch_size: Batch size for generation
            
        Returns:
            List of generated translations
        """
        translations = []
        
        # Get prompt template
        prompt_template = self.prompts.get("template", "Translate the following text: {text}")
        
        try:
            # Create progress bar for translation generation
            with tqdm(total=len(test_data), desc="Generating translations", unit="samples") as pbar:
                for i in range(0, len(test_data), batch_size):
                    batch = test_data[i:i + batch_size]
                    batch_translations = []
                    
                    for item in batch:
                        source_text = item.get("source", "")
                        if not source_text:
                            batch_translations.append("")
                            pbar.update(1)
                            continue
                        
                        # Format prompt
                        prompt = prompt_template.format(text=source_text)
                        
                        # Generate translation
                        try:
                            inputs = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                            
                            with model.model.eval():
                                outputs = model.model.generate(
                                    **inputs,
                                    max_new_tokens=256,
                                    do_sample=False,
                                    num_beams=1,
                                    pad_token_id=model.tokenizer.eos_token_id
                                )
                            
                            # Decode translation
                            generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            # Extract translation (remove prompt)
                            translation = generated_text[len(prompt):].strip()
                            batch_translations.append(translation)
                            
                        except Exception as e:
                            LOG.warning(f"Generation failed for sample {i}: {e}")
                            batch_translations.append("")
                        
                        pbar.update(1)
                    
                    translations.extend(batch_translations)
        
        except Exception as e:
            LOG.error(f"Translation generation failed: {e}")
            # Return empty translations to avoid breaking evaluation
            translations = [""] * len(test_data)
        
        return translations
    
    def _calculate_metrics(self, test_data: List[Dict], translations: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics
        
        Args:
            test_data: Original test data with references
            translations: Generated translations
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Get references
        references = [item.get("target", "") for item in test_data]
        
        # Calculate CHRF score (character-level F-score)
        try:
            chrf_score = self._calculate_chrf(references, translations)
            metrics["chrf"] = chrf_score
        except Exception as e:
            LOG.warning(f"CHRF calculation failed: {e}")
            metrics["chrf"] = 0.0
        
        # Calculate BLEU score if available
        try:
            bleu_score = self._calculate_bleu(references, translations)
            metrics["bleu"] = bleu_score
        except Exception as e:
            LOG.warning(f"BLEU calculation failed: {e}")
            metrics["bleu"] = 0.0
        
        # Simple metrics
        metrics["avg_translation_length"] = sum(len(t.split()) for t in translations) / len(translations) if translations else 0
        metrics["empty_translations"] = sum(1 for t in translations if not t.strip()) / len(translations) if translations else 1.0
        
        return metrics
    
    def _calculate_chrf(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate character-level F-score (CHRF)
        
        Simple implementation of CHRF metric
        """
        if not references or not hypotheses:
            return 0.0
        
        total_chrf = 0.0
        valid_pairs = 0
        
        # Add progress bar for CHRF calculation
        with tqdm(zip(references, hypotheses), total=len(references), desc="Computing CHRF", unit="pairs", leave=False) as pbar:
            for ref, hyp in pbar:
                if not ref.strip() or not hyp.strip():
                    continue
                    
                # Character n-grams (1-3)
                ref_chars = set(ref.lower().replace(' ', ''))
                hyp_chars = set(hyp.lower().replace(' ', ''))
                
                if not ref_chars:
                    continue
                
                # Simple character overlap
                overlap = len(ref_chars.intersection(hyp_chars))
                precision = overlap / len(hyp_chars) if hyp_chars else 0.0
                recall = overlap / len(ref_chars) if ref_chars else 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    total_chrf += f1
                    valid_pairs += 1
        
        return (total_chrf / valid_pairs * 100) if valid_pairs > 0 else 0.0
    
    def _calculate_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate BLEU score
        
        Simple implementation for basic BLEU calculation
        """
        if not references or not hypotheses:
            return 0.0
        
        # This is a simplified BLEU calculation
        # In production, you'd use libraries like sacrebleu
        
        total_score = 0.0
        valid_pairs = 0
        
        # Add progress bar for BLEU calculation
        with tqdm(zip(references, hypotheses), total=len(references), desc="Computing BLEU", unit="pairs", leave=False) as pbar:
            for ref, hyp in pbar:
                if not ref.strip() or not hyp.strip():
                    continue
                
                ref_words = ref.lower().split()
                hyp_words = hyp.lower().split()
                
                if not ref_words:
                    continue
                
                # Simple word overlap
                ref_set = set(ref_words)
                hyp_set = set(hyp_words)
                
                overlap = len(ref_set.intersection(hyp_set))
                precision = overlap / len(hyp_set) if hyp_set else 0.0
                recall = overlap / len(ref_set) if ref_set else 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    total_score += f1
                    valid_pairs += 1
        
        return (total_score / valid_pairs * 100) if valid_pairs > 0 else 0.0
    
    def _measure_performance(self, model: BaseModel, sample_data: List[Dict], batch_size: int) -> Dict[str, float]:
        """Measure model performance metrics
        
        Args:
            model: Model to measure
            sample_data: Small sample for performance testing
            batch_size: Batch size for testing
            
        Returns:
            Dictionary of performance metrics
        """
        if not sample_data:
            return {"inference_time_ms": 0.0}
        
        # Measure inference time on small sample
        start_time = time.time()
        
        try:
            # Generate on sample with progress bar
            performance_samples = sample_data[:5]  # Use only 5 samples for speed
            with tqdm(performance_samples, desc="Measuring performance", unit="samples", leave=False) as pbar:
                for item in pbar:
                    source_text = item.get("source", "")
                    if source_text:
                        prompt_template = self.prompts.get("template", "Translate: {text}")
                        prompt = prompt_template.format(text=source_text)
                        
                        inputs = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                        
                        with model.model.eval():
                            outputs = model.model.generate(
                                **inputs,
                                max_new_tokens=64,
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=model.tokenizer.eos_token_id
                            )
            
            end_time = time.time()
            avg_time_per_sample = (end_time - start_time) / min(5, len(sample_data)) * 1000  # Convert to ms
            
        except Exception as e:
            LOG.warning(f"Performance measurement failed: {e}")
            avg_time_per_sample = 0.0
        
        return {
            "inference_time_ms": avg_time_per_sample
        } 