import sys
from pathlib import Path
import argparse
import logging
import torch
import os

sys.path.append(str(Path(__file__).parent))

from segmentation.mask_rcnn import MaskRCNNSegmenter
from segmentation.advanced_segmentation import AdvancedMaskRCNN, UNetSegmenter, AttentionUNet, EnsembleSegmenter
from segmentation.transformer_segmentation import VisionTransformerSegmentation, SegFormer
from segmentation.deep_lab import DeepLabV3Plus, LRASPP, FCN
from segmentation.advanced_unet import DenseUNet, ResUNet, MultiScaleUNet
from segmentation.panoptic_segmentation import PanopticFPN, PanopticDeepLab
from segmentation.postprocessing import SegmentationPostProcessor
from llm.report_generator import LLaMA2ReportGenerator
from llm.advanced_llm import MultiLLMEnsemble, LoRAFineTuner
from llm.prompt_optimization import PromptOptimizer, FewShotPromptBuilder, ChainOfThoughtPrompting
from training.segmentation_training import AdvancedSegmentationTrainer
from training.advanced_training import AdvancedSegmentationTrainer as SophisticatedTrainer
from data.medical_data_loader import MedicalDataManager
from data.preprocessing import MedicalImagePreprocessor
from evaluation.segmentation_metrics import SegmentationEvaluator
from evaluation.comprehensive_metrics import ComprehensiveSegmentationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(config: dict, data_path: str):
    logger.info("=" * 80)
    logger.info("TRAINING MEDSEG-LLM MODELS")
    logger.info("=" * 80)
    
    data_manager = MedicalDataManager(config)
    
    if not os.path.exists(data_path):
        logger.warning(f"Data path not found: {data_path}")
        logger.info("Please provide a valid data directory")
        return
    
    image_paths, mask_paths = data_manager.load_from_directory(data_path, mask_dir=os.path.join(data_path, 'masks'))
    
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        batch_size=config.get('batch_size', 4)
    )
    
    segmenter = MultiScaleUNet(in_channels=3, num_classes=2)
    
    trainer = SophisticatedTrainer(
        model=segmenter,
        config=config.get('training', {}),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, config.get('num_epochs', 10))
    
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Metrics: {test_metrics}")
    
    evaluator = ComprehensiveSegmentationEvaluator()
    all_preds = []
    all_targets = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(trainer.device)
            masks = batch['mask'].to(trainer.device)
            outputs = trainer.model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            preds = outputs.argmax(dim=1)
            
            all_preds.extend([p.cpu().numpy() for p in preds])
            all_targets.extend([m.cpu().numpy() for m in masks])
    
    comprehensive_metrics = evaluator.evaluate_comprehensive(all_preds, all_targets)
    logger.info(f"Comprehensive Metrics: {comprehensive_metrics}")
    
    model_save_path = 'models/best_segmentation_model.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(trainer.model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


def generate_report(scan_path: str, findings: list, model_path: str = None):
    logger.info("Generating diagnostic report...")
    
    report_generator = LLaMA2ReportGenerator()
    
    if model_path and os.path.exists(model_path):
        report_generator.load_model(model_path)
    
    report = report_generator.generate_structured_report(findings)
    
    output_path = 'reports/diagnostic_report.json'
    os.makedirs('reports', exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")
    return report


def start_api():
    logger.info("Starting FastAPI server...")
    from api.fastapi_server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    parser = argparse.ArgumentParser(description='MedSeg-LLM')
    parser.add_argument('--mode', type=str, choices=['train', 'segment', 'generate', 'api'], default='train')
    parser.add_argument('--data', type=str, default='data/scans')
    parser.add_argument('--scan', type=str)
    parser.add_argument('--findings', type=str)
    parser.add_argument('--model', type=str, default='models/best_segmentation_model.pt')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    
    args = parser.parse_args()
    
    config = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("MEDSEG-LLM - MEDICAL SEGMENTATION & REPORT GENERATION")
    logger.info("=" * 80)
    
    if args.mode == 'train':
        train_models(config, args.data)
    elif args.mode == 'segment' and args.scan:
        segmenter = MaskRCNNSegmenter()
        masks, scores, boxes = segmenter.segment(args.scan)
        logger.info(f"Segmented {len(masks)} regions")
    elif args.mode == 'generate' and args.findings:
        import json
        findings = json.loads(args.findings) if args.findings.startswith('[') else [{'description': args.findings}]
        report = generate_report(None, findings)
        logger.info("Report generated")
    elif args.mode == 'api':
        start_api()


if __name__ == '__main__':
    main()
