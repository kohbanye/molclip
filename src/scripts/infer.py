from molclip.config import DataConfig, MolClipConfig, MolConfig, TextConfig, TrainConfig
from molclip.feature import get_mol_graph
from molclip.model import MolClip


def main():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    texts = [
        "aspirin",  # correct
        "This molecule is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation",  # correct
        "This molecule appears as a clear, colorless liquid with a characteristic aromatic odor.",  # incorrect
    ]

    config = MolClipConfig(
        data=DataConfig(),
        text=TextConfig(),
        mol=MolConfig(),
        train=TrainConfig(),
    )

    ckpt_path = "molclip/y4cau8oc/checkpoints/epoch=29-step=24780.ckpt"
    model = MolClip.load_from_checkpoint(ckpt_path, config=config)

    mol_graph = get_mol_graph(smiles)
    mol_embed = model.mol_encoder(mol_graph.cuda())

    text_embeds = []
    tokenizer = model.get_tokenizer()
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        text_embed = model.text_encoder.forward(input_ids.cuda())  # type: ignore
        text_embeds.append(text_embed)

    print(f"Molecule: {smiles}")
    for text, text_embed in zip(texts, text_embeds):
        similarity = text_embed @ mol_embed.T
        print(f"Text: {text}\nSimilarity: {similarity.item()}\n")


if __name__ == "__main__":
    main()
