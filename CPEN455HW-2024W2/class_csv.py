import csv
import argparse
import torch
from torchvision import transforms
from tqdm import tqdm
import os
from utils import *
from dataset import *
from model import *
from classification_evaluation import *

NUM_CLASSES = len(my_bidict)

def get_label(model, model_input, device):
    B = model_input.size(0)
    model_input = model_input.to(device)

    a_log_likelihood = []
    for nr_label in range(NUM_CLASSES):
        labels = torch.full((B,), nr_label, dtype=torch.long).to(device)
        out = model(model_input, labels)
        log_likelihood = -discretized_mix_logistic_loss(model_input, out, Bayes=True)
        a_log_likelihood.append(log_likelihood.view(-1, 1))

    log_likelihood = torch.cat(a_log_likelihood, dim=1)
    return torch.argmax(log_likelihood, dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='data', help='Dataset root')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-m', '--mode', type=str, default='test', help='Should be "test" for this script')
    parser.add_argument('-o', '--output_csv', type=str, default='test_submission.csv', help='Output CSV file')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/CPEN455/models_7_film_mid_late_noaugdrop_1/epoch_130.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': False}

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        rescaling
    ])

    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=transform),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=160, input_channels=3, nr_logistic_mix=5,
                     film=True, mid_fusion=True, late_fusion=True).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"❌ Model not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    predictions = []
    for model_input, paths in tqdm(dataloader):
        model_input = model_input.to(device)
        outputs = get_label(model, model_input, device)

        for path, pred in zip(paths, outputs):
            rel_path = path.replace(args.data_dir + '/', '')  # ex: test/0000012.jpg
            predictions.append([rel_path, pred.item()])

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)

    print(f"✅ Saved {len(predictions)} predictions to {args.output_csv}")
