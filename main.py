import argparse
from src.dataset.generate_dataset import generate_dataset
from src.model.train import train_model
from src.model.test import test_model
from src.fairness.evaluate import evaluate_fairness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Welcome to Quantifying Bias in Parkinson Models")
    parser.add_argument("--generate-data", action="store_true", help="Generate necessary datasets from raw data")
    parser.add_argument("--train-model", action="store_true", help="Train the model on featureset.")
    parser.add_argument("--test-model", action="store_true", help="Test the model on held out test data.")
    parser.add_argument("--evaluate-model-bias", type=str, help="Evaluate bias using a protected attribute (Takes race_upsample_true or race_upsample_false)")

    args = parser.parse_args()

    if args.generate_data:
        generate_dataset()

    elif args.train_model:
        train_model()

    elif args.test_model:
        test_model()

    elif args.evaluate_model_bias:
        race_upsample = args.evaluate_model_bias
        
        if race_upsample not in ["race_upsample_true", "race_upsample_false"]:
            print("❌ Invalid option. Use 'race_upsample_true' or 'race_upsample_false'.")
            exit(1)
        
        race_upsample_flag = race_upsample.split('_')[-1].lower() == 'true'
        evaluate_fairness(race_upsample=race_upsample_flag)

    else:
        print("Wrong Input")
