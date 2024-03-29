def main():
    import argparse, sys, os

    if len(sys.argv) == 1:
        sys.argv.append('-h')
    
    epilog = "\n" + \
             "Examples: " + \
             "\n" + \
             " Train a model:\n" + \
             "   biodeg train -e 100 -i data/data/All-Public_dataset_Mordred.csv -o data/models/train_epoch100_all_dataset.pt\n" + \
             "\n" + \
             " Test model accuracy:\n" + \
             "   biodeg test -m data/models/train_epoch100_all_dataset.pt -i data/data/All-Public_dataset_Mordred.csv\n" + \
             "\n" + \
             " Make prediction on real samples:\n" + \
             "   biodeg predict -i data/data/All-Public_dataset_Mordred_tail_10.csv -m data/models/train_epoch100_all_dataset.pt\n";

    def valid_file(param):
        if not os.path.exists(param):
            print(f"The file {param} does not exist.")
            sys.exit(1)
        return param
        
    def model_path(file):
        return os.path.join(os.path.dirname(__file__), "data", "models", file)

    parser = argparse.ArgumentParser(description="Evaluate biodegradability of chemical compounds",
        epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    parser_train = subparsers.add_parser("train", help="Train model")
    parser_train.add_argument("-e", "--epoch", type=int, default=10, help="Number of epoch for the training step")
    parser_train.add_argument("-o", "--output", type=str, default="model.pt", help="Output file for the model")
    parser_train.add_argument("-i", "--input", type=valid_file, help="Input file to train the model with")
    parser_train.add_argument("-t", "--test", action="store_true", default=False, help="Run tests after model training")

    parser_predict = subparsers.add_parser("predict", help="Predict biodegradability")
    parser_predict.add_argument("-i", "--input", type=valid_file, help="Input file to evaluate")
    parser_predict.add_argument("-s","--smiles", type=str,help="Make prediction on given SMILES")
    parser_predict.add_argument("-m", "--model", default=model_path("model.pt"), type=valid_file, help="Model to use to make the prediction")
    parser_predict.add_argument("-o", "--output", default="guessed.csv", type=str, help="Where to put the results")

    parser_predict = subparsers.add_parser("test", help="Test a model on some data")
    parser_predict.add_argument("-i", "--input", type=valid_file, help="Input file to evaluate")
    parser_predict.add_argument("-m", "--model", default=model_path("model.pt"), type=valid_file, help="Model to use to make the prediction")

    args = parser.parse_args()

    from . import BioDegClassifier
    
    if args.command == "train":
        d = BioDegClassifier.Dev()
        d.loadData(args.input)
        d.train(args.epoch)
        d.save(args.output)
        if args.test:
            print('Total accuracy on this data %.4f' % d.test())
    elif args.command == "predict":
        c = BioDegClassifier.Prod()
        if args.input:
            c.loadData(args.input)
        else:
            c.loadMols([Chem.MolFromSmiles(args.smiles)]) 
    
        c.load(args.model)
        result = c.guess()
        c.guess_result_to_csv(args.output,result)
        print('Result as %s' % args.output)
    elif args.command == "test":
        d = BioDegClassifier.Dev()
        d.loadData(args.input)
        d.load(args.model)
        print('Total accuracy on this data %.4f' % d.test())

if __name__ == "__main__":
    main()
else:
    from . import BioDegClassifier
    from .BioDegClassifierModel import *
