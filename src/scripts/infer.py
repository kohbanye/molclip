from molclip.config import DataConfig, MolClipConfig, MolConfig, TextConfig, TrainConfig
from molclip.feature import get_mol_graph
from molclip.model import MolClip
from molclip.utils import fix_seeds


def main(ckpt_path: str) -> None:
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    texts = [
        "This molecule is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation.",  # correct, not in the dataset
        "A non-steroidal anti-inflammatory drug with cyclooxygenase inhibitor activity.",  # correct, from the dataset
        "This molecule appears as a clear, colorless liquid with a characteristic aromatic odor.",  # incorrect
        "This is a proton-pump inhibitor, used to treat gastric acid-related disorders.",  # incorrect
        "選択的セロトニン再取り込み阻害薬（SSRI）に分類される抗うつ薬。",  # incorrect
    ]

    config = MolClipConfig(
        data=DataConfig(),
        text=TextConfig(),
        mol=MolConfig(),
        train=TrainConfig(),
    )

    fix_seeds()
    model = MolClip.load_from_checkpoint(ckpt_path, config=config)
    # model = MolClip(config).cuda()

    mol_graph = get_mol_graph(smiles)
    mol_embed = model.mol_encoder(mol_graph.cuda())

    text_embeds = []
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer(texts, padding="max_length", max_length=config.data.max_length, return_tensors="pt")[
        "input_ids"
    ]
    text_embeds = model.text_encoder.forward(input_ids.cuda())  # type: ignore

    logits = (mol_embed @ text_embeds.T).softmax(dim=-1)[0]

    print(f"Molecule: {smiles}\n")
    for text, logit in zip(texts, logits):
        print(f"Text: {text}")
        print(f"Probability: {logit.item() * 100:.3f}%\n")


if __name__ == "__main__":
    ckpt_path = "molclip/0.1.1/checkpoints/epoch=29-step=24780.ckpt"
    main(ckpt_path)
